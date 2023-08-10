import argparse
import hashlib
import logging
import lzma
import tamp
import readline
import struct
import sys
from pyocd.core.exceptions import ProbeError
from pyocd.coresight.coresight_target import CoreSightTarget
from pyocd.core.memory_map import (FlashRegion, RamRegion, MemoryMap)
from pyocd.coresight.cortex_m import CortexM
from pyocd.coresight.minimal_mem_ap import MinimalMemAP
import usb
import shlex
from copy import copy
from pathlib import Path
from time import sleep, time
from typing import Union, List, Optional
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from datetime import datetime, timezone
from PIL import Image
from copy import deepcopy

from collections import namedtuple

from pyocd.core.memory_map import FlashRegion
from pyocd.core.helpers import ConnectHelper
from pyocd.flash.eraser import FlashEraser
from pyocd.flash.file_programmer import FileProgrammer

from littlefs import LittleFS, LittleFSError


logging.getLogger('pyocd').setLevel(logging.WARNING)


context_counter = 1
sleep_duration = 0.05


def sha256(data):
    return hashlib.sha256(data).digest()


_EMPTY_HASH_DIGEST = sha256(b"")
Variable = namedtuple("Variable", ['address', 'size'])

# fmt: off
# These addresses are fixed via a carefully crafted linker script.
comm = {
    "framebuffer1":            Variable(0x2400_0000, 320 * 240 * 2),
    "framebuffer2":            Variable(0x2402_5800, 320 * 240 * 2),
    "boot_magic":              Variable(0x2000_0000, 4),
    "log_idx":                 Variable(0x2000_0008, 4),
    "logbuf":                  Variable(0x2000_000c, 4096),
    "lfs_cfg":                 Variable(0x2000_1010, 4),
}
contexts = [{} for i in range(2)]

def _populate_comm():
    # Communication Variables; put in a function to prevent variable leakage.
    comm["flashapp_comm"] = comm["framebuffer2"]

    comm["flashapp_state"]           = last_variable = Variable(comm["flashapp_comm"].address, 4)
    comm["program_status"]           = last_variable = Variable(last_variable.address + last_variable.size, 4)
    comm["utc_timestamp"]            = last_variable = Variable(last_variable.address + last_variable.size, 4)
    comm["program_chunk_idx"]        = last_variable = Variable(last_variable.address + last_variable.size, 4)
    comm["program_chunk_count"]      = last_variable = Variable(last_variable.address + last_variable.size, 4)
    comm["active_context_index"]     = last_variable = Variable(last_variable.address + last_variable.size, 4)


    for i in range(2):
        struct_start = comm["flashapp_comm"].address + ((i+1)*4096)
        contexts[i]["ready"]             = last_variable = Variable(struct_start, 4)
        contexts[i]["size"]              = last_variable = Variable(last_variable.address + last_variable.size, 4)
        contexts[i]["address"]           = last_variable = Variable(last_variable.address + last_variable.size, 4)
        contexts[i]["erase"]             = last_variable = Variable(last_variable.address + last_variable.size, 4)
        contexts[i]["erase_bytes"]       = last_variable = Variable(last_variable.address + last_variable.size, 4)
        contexts[i]["decompressed_size"] = last_variable = Variable(last_variable.address + last_variable.size, 4)
        contexts[i]["expected_sha256"]   = last_variable = Variable(last_variable.address + last_variable.size, 32)
        contexts[i]["expected_sha256_decompressed"]   = last_variable = Variable(last_variable.address + last_variable.size, 32)

        # Don't ever directly use this, just here for alignment purposes
        contexts[i]["__buffer_ptr"]        = last_variable = Variable(last_variable.address + last_variable.size, 4)

    struct_start = comm["flashapp_comm"].address + (3*4096)
    comm["active_context"] = last_variable = Variable(struct_start, 4096)

    for i in range(2):
        contexts[i]["buffer"]            = last_variable = Variable(last_variable.address + last_variable.size, 256 << 10)

    comm["decompress_buffer"] = last_variable = Variable(last_variable.address + last_variable.size, 256 << 10)

    # littlefs config struct elements
    comm["lfs_cfg_context"]      = Variable(comm["lfs_cfg"].address + 0,  4)
    comm["lfs_cfg_read"]         = Variable(comm["lfs_cfg"].address + 4,  4)
    comm["lfs_cfg_prog"]         = Variable(comm["lfs_cfg"].address + 8,  4)
    comm["lfs_cfg_erase"]        = Variable(comm["lfs_cfg"].address + 12, 4)
    comm["lfs_cfg_sync"]         = Variable(comm["lfs_cfg"].address + 16, 4)
    comm["lfs_cfg_read_size"]    = Variable(comm["lfs_cfg"].address + 20, 4)
    comm["lfs_cfg_prog_size"]    = Variable(comm["lfs_cfg"].address + 24, 4)
    comm["lfs_cfg_block_size"]   = Variable(comm["lfs_cfg"].address + 28, 4)
    comm["lfs_cfg_block_count"]  = Variable(comm["lfs_cfg"].address + 32, 4)
    # TODO: too lazy to add the other lfs_config attributes

_populate_comm()


_flashapp_state_enum_to_str = {
    0x00000000: "INIT",
    0x00000001: "IDLE",
    0x00000002: "START",
    0x00000003: "CHECK_HASH_RAM_NEXT",
    0x00000004: "CHECK_HASH_RAM",
    0x00000005: "DECOMPRESSING",
    0x00000006: "ERASE_NEXT",
    0x00000007: "ERASE",
    0x00000008: "PROGRAM_NEXT",
    0x00000009: "PROGRAM",
    0x0000000a: "CHECK_HASH_FLASH_NEXT",
    0x0000000b: "CHECK_HASH_FLASH",
    0x0000000c: "FINAL",
    0x0000000d: "ERROR",
}
_flashapp_state_str_to_enum = {v: k for k, v in _flashapp_state_enum_to_str.items()}

_flashapp_status_enum_to_str  = {
    0         : "BOOTING",
    0xbad00001: "BAD_HASH_RAM",
    0xbad00002: "BAD_HAS_FLASH",
    0xbad00003: "NOT_ALIGNED",
    0xcafe0000: "IDLE",
    0xcafe0001: "DONE",
    0xcafe0002: "BUSY",
}
_flashapp_status_str_to_enum = {v: k for k, v in _flashapp_status_enum_to_str.items()}

# fmt: on


##############
# Exceptions #
##############

class TimeoutError(Exception):
    """Some operation timed out."""


class DataError(Exception):
    """Some data was not as expected."""


class StateError(Exception):
    """On-device flashapp is in the ERROR state."""


###############
# Compression #
###############
def compress_lzma(data):
    compressed_data = lzma.compress(
        data,
        format=lzma.FORMAT_ALONE,
        filters=[
            {
                "id": lzma.FILTER_LZMA1,
                "preset": 6,
                "dict_size": 16 * 1024,
            }
        ],
    )

    return compressed_data[13:]


def compress_chunks(chunks: List[bytes], max_workers=2):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(compress_lzma, chunk) for chunk in chunks]
        for future in futures:
            yield future.result()


############
# LittleFS #
############
class LfsDriverContext:
    def __init__(self, offset) -> None:
        validate_extflash_offset(offset)
        self.offset = offset
        self.cache = {}

    def read(self, cfg: 'LFSConfig', block: int, off: int, size: int) -> bytes:
        logging.getLogger(__name__).debug('LFS Read : Block: %d, Offset: %d, Size=%d' % (block, off, size))
        try:
            return bytes(self.cache[block][off:off+size])
        except KeyError:
            pass
        wait_for_all_contexts_complete()  # if a prog/erase is being performed, chip is not in memory-mapped-mode
        self.cache[block] = bytearray(extflash_read(self.offset + (block * cfg.block_size), size))
        return bytes(self.cache[block][off:off+size])

    def prog(self, cfg: 'LFSConfig', block: int, off: int, data: bytes) -> int:
        logging.getLogger(__name__).debug('LFS Prog : Block: %d, Offset: %d, Data=%r' % (block, off, data))

        # Update the local block if it has previosly been read
        try:
            barray = self.cache[block]
            barray[off:off + len(data)] = data
        except KeyError:
            pass

        decompressed_hash = sha256(data)
        compressed_data = compress_lzma(data)

        extflash_write(self.offset + (block * cfg.block_size) + off,
                       compressed_data,
                       erase=False,
                       decompressed_size=len(data),
                       decompressed_hash=decompressed_hash,
                       )
        return 0

    def erase(self, cfg: 'LFSConfig', block: int) -> int:
        logging.getLogger(__name__).debug('LFS Erase: Block: %d' % block)
        self.cache[block] = bytearray([0xFF]*cfg.block_size)
        extflash_erase(self.offset + (block * cfg.block_size), cfg.block_size)
        return 0

    def sync(self, cfg: 'LFSConfig') -> int:
        return 0

def is_existing_gnw_dir(fs, path: Union[str, Path]):
    if isinstance(path, Path):
        path = path.as_posix()

    try:
        stat = fs.stat(path)
    except LittleFSError as e:
        if e.code == -2:  # LFS_ERR_NOENT
            return False
        raise
    return stat.type == 2


def gnw_sha256(fs, path: Union[str, Path]):
    if isinstance(path, Path):
        path = path.as_posix()

    try:
        with fs.open(path, "rb") as f:
            data = f.read()
    except FileNotFoundError:
        return bytes(32)

    return sha256(data)



def timestamp_now() -> int:
    return int(round(datetime.now().replace(tzinfo=timezone.utc).timestamp()))


def timestamp_now_bytes() -> bytes:
    return timestamp_now().to_bytes(4, "little")


#########
# PyOCD #
#########
class DBGMCU:
    CR = 0xE00E1004
    CR_VALUE = 0x7 # DBG_STANDBY | DBG_STOP | DBG_SLEEP

FLASH_ALGO = {
    'load_address' : 0x20000000,

    # Flash algorithm as a hex string
    'instructions': [
    0xe7fdbe00,
    0x4770ba40, 0x4770bac0, 0x0030ea4f, 0x00004770, 0x8f4ff3bf, 0xb5104770, 0x48ea4603, 0x61604cea,
    0x48e9bf00, 0xf0006900, 0x28000001, 0x48e7d1f9, 0x60604ce5, 0x606048e6, 0x4ce648e2, 0xbf006020,
    0x1f0048e4, 0xf0006800, 0x28000001, 0x48dfd1f8, 0x3c104ce0, 0x48de6020, 0xf8c44cdb, 0x46200104,
    0x200069c0, 0x4601bd10, 0x68c048d7, 0x0001f040, 0x60d04ad5, 0x380848d7, 0xf0406800, 0xf8c20001,
    0x2000010c, 0xbf004770, 0x690048cf, 0x0001f000, 0xd1f92800, 0x49cc48cb, 0x46086148, 0xf02068c0,
    0x60c80001, 0x68c04608, 0x0008f040, 0x460860c8, 0xf04068c0, 0x60c80020, 0x48c3bf00, 0xf0006900,
    0x28000001, 0x48c0d1f9, 0xf02068c0, 0x49be0008, 0xbf0060c8, 0x1f0048bf, 0xf0006800, 0x28000001,
    0x48b8d1f8, 0x600849bb, 0xf8d048b7, 0xf020010c, 0x49b50001, 0x010cf8c1, 0xf8d04608, 0xf040010c,
    0xf8c10008, 0x4608010c, 0x010cf8d0, 0x0020f040, 0x390849b0, 0xbf006008, 0x1f0048ae, 0xf0006800,
    0x28000001, 0x48abd1f8, 0x68003808, 0x0008f020, 0xf8c149a5, 0x2000010c, 0xb5104770, 0xf3c14601,
    0xf1b13247, 0xd3366f00, 0x6f01f1b1, 0x489ed233, 0x4ba16940, 0x4b9c4318, 0xbf006158, 0x6900489a,
    0x0004f000, 0xd1f92800, 0x68c04897, 0x50fef420, 0x60d84b95, 0x68c04618, 0xea432304, 0x43181382,
    0x60d84b91, 0x68c04618, 0x0020f040, 0xbf0060d8, 0x6900488d, 0x0004f000, 0xd1f92800, 0x68c0488a,
    0x0004f020, 0x60d84b88, 0x69004618, 0x0001f000, 0x2001b3f0, 0x4887bd10, 0x4b876800, 0x4b824318,
    0x0114f8c3, 0x4883bf00, 0x68001f00, 0x0004f000, 0xd1f82800, 0x3808487f, 0xf4206800, 0x4b7a50fe,
    0x010cf8c3, 0x3808487b, 0xf1a26800, 0x24040380, 0x1383ea44, 0x4b744318, 0x010cf8c3, 0xf8d04618,
    0xf040010c, 0xf8c30020, 0xbf00010c, 0x1f004871, 0xf0006800, 0x28000004, 0x486ed1f8, 0x68003808,
    0x0004f020, 0xf8c34b68, 0x486a010c, 0xe0001f00, 0x6800e005, 0x0001f000, 0x2001b108, 0xf7ffe7ba,
    0x2000fee7, 0xb5f0e7b6, 0x46164603, 0x4635461a, 0xbf002400, 0x6900485c, 0x0001f000, 0xd1f92800,
    0x4f594858, 0xbf006178, 0x1f00485a, 0xf0006800, 0x28000001, 0x4853d1f8, 0x60384f56, 0x4852e09c,
    0xf02068c0, 0x4f500001, 0x463860f8, 0xf04068c0, 0x60f80002, 0x3808484f, 0xf0206800, 0xf8c70001,
    0x4638010c, 0x010cf8d0, 0x0002f040, 0x010cf8c7, 0xd30c2910, 0xe0062400, 0x6868682f, 0x60506017,
    0x32083508, 0x2c021c64, 0x3910dbf6, 0x2400e028, 0xf815e004, 0xf8020b01, 0x1c640b01, 0xd3f8428c,
    0xe0032400, 0xf80220ff, 0x1c640b01, 0x0010f1c1, 0xd8f742a0, 0x6f00f1b3, 0xf1b3d309, 0xd2066f01,
    0x68c04831, 0x0040f040, 0x60f84f2f, 0x4831e007, 0x68003808, 0x0040f040, 0xf8c74f2b, 0x2100010c,
    0xfe76f7ff, 0x6f00f1b3, 0xf1b3d30a, 0xd2076f01, 0x4825bf00, 0xf0006900, 0x28000001, 0xe007d1f9,
    0x4824bf00, 0x68001f00, 0x0001f000, 0xd1f82800, 0x6900481d, 0x4f1f2000, 0x683f1f3f, 0xb1b04300,
    0x6f00f1b3, 0xf1b3d309, 0xd2066f01, 0x68c04816, 0x0002f020, 0x60f84f14, 0x4816e007, 0x68003808,
    0x0002f020, 0xf8c74f10, 0x2001010c, 0xf1b3bdf0, 0xd3096f00, 0x6f01f1b3, 0x480bd206, 0xf02068c0,
    0x4f090002, 0xe00760f8, 0x3808480a, 0xf0206800, 0x4f050002, 0x010cf8c7, 0xf47f2900, 0x2000af60,
    0x0000e7e4, 0x0faf0000, 0x52002000, 0x45670123, 0xcdef89ab, 0x52002114, 0x0fef0000, 0x00000000
    ],

    # Relative function addresses
    'pc_init': 0x2000001b,
    'pc_unInit': 0x2000006b,
    'pc_program_page': 0x2000024b,
    'pc_erase_sector': 0x2000013f,
    'pc_eraseAll': 0x2000008b,

    'static_base' : 0x20000000 + 0x00000004 + 0x000003dc,
    'begin_stack' : 0x200113f0,
    'end_stack' : 0x200103f0,
    'begin_data' : 0x20000000 + 0x1000,
    'page_size' : 0x8000,
    'analyzer_supported' : False,
    'analyzer_address' : 0x00000000,
    # Enable double buffering
    'page_buffers' : [
        0x200003f0,
        0x200083f0
    ],
    'min_program_length' : 0x8000,

    # Relative region addresses and sizes
    'ro_start': 0x4,
    'ro_size': 0x3dc,
    'rw_start': 0x3e0,
    'rw_size': 0x4,
    'zi_start': 0x3e4,
    'zi_size': 0x0,

    # Flash information
    'flash_start': 0x8000000,
    'flash_size': 0x40000,
    'sector_sizes': (
        (0x0, 0x2000),
    )
}

class STM32H7B0_256K(CoreSightTarget):

    VENDOR = "STMicroelectronics"

    MEMORY_MAP = MemoryMap(
        FlashRegion( start=0x08000000, length=0x40000,  sector_size=0x2000,
                                                        page_size=0x8000,
                                                        is_boot_memory=True,
                                                        algo=FLASH_ALGO,
                                                        name="bank_1"),
        FlashRegion( start=0x08100000, length=0x40000,  sector_size=0x2000,
                                                        page_size=0x8000,
                                                        algo=FLASH_ALGO,
                                                        name="bank_2"),
        RamRegion(   start=0x20000000, length=0x20000, name="dtcm"),
        RamRegion(   start=0x24000000, length=0x40000, name="axi_sram_1"),
        RamRegion(   start=0x24040000, length=0x60000, name="axi_sram_2"),
        RamRegion(   start=0x240A0000, length=0x60000, name="axi_sram_3"),
        RamRegion(   start=0x30000000, length=0x10000, name="ahb_sram_1"),
        RamRegion(   start=0x30010000, length=0x10000, name="ahb_sram_2"),
        RamRegion(   start=0x38000000, length=0x8000, name="sdr_sram"),
        RamRegion(   start=0x38800000, length=0x1000, name="backup_sram"),
        )

    def __init__(self, session):
        super().__init__(session, self.MEMORY_MAP)

    def assert_reset_for_connect(self):
        self.dp.assert_reset(1)

    def safe_reset_and_halt(self):
        assert self.dp.is_reset_asserted()

        # At this point we can't access full AP as it is not initialized yet.
        # Let's create a minimalistic AP and use it.
        ap = MinimalMemAP(self.dp)
        ap.init()

        DEMCR_value = ap.read32(CortexM.DEMCR)

        # Halt on reset.
        ap.write32(CortexM.DEMCR, CortexM.DEMCR_VC_CORERESET)
        ap.write32(CortexM.DHCSR, CortexM.DBGKEY | CortexM.C_DEBUGEN)

        # Prevent disabling bus clock/power in low power modes.
        ap.write32(DBGMCU.CR, DBGMCU.CR_VALUE)

        self.dp.assert_reset(0)
        sleep(0.01)

        # Restore DEMCR original value.
        ap.write32(CortexM.DEMCR, DEMCR_value)

    def create_init_sequence(self):
        # STM32 under some low power/broken clock states doesn't allow AHP communication.
        # Low power modes are quite popular on stm32 (including MBed OS defaults).
        # 'attach' mode is broken by default, as STM32 can't be connected on low-power mode
        #  successfully without previous DBGMCU setup (It is not possible to write DBGMCU).
        # It is also not possible to run full pyOCD discovery code under-reset.
        #
        # As a solution we can setup DBGMCU under reset, halt core and release reset.
        # Unfortunately this code has to be executed _before_ discovery stage
        # and without discovery stage we don't have access to AP/Core.
        # As a solution we can create minimalistic AP implementation and use it
        # to setup core halt.
        # So the sequence for 'halt' connect mode will look like
        # -> Assert reset
        # -> Connect DebugPort
        # -> Setup MiniAp
        # -> Setup halt on reset
        # -> Enable support for debugging in low-power modes
        # -> Release reset
        # -> [Core is halted and reset is released]
        # -> Continue [discovery, create cores, etc]
        seq = super().create_init_sequence()
        if self.session.options.get('connect_mode') in ('halt', 'under-reset'):
            seq.insert_before('dp_init', ('assert_reset_for_connect', self.assert_reset_for_connect))
            seq.insert_after('dp_init', ('safe_reset_and_halt', self.safe_reset_and_halt))

        return seq

    def post_connect_hook(self):
        self.write32(DBGMCU.CR, DBGMCU.CR_VALUE)


def chunk_bytes(data, chunk_size):
    return [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]


def read_int(key: Union[int, str, Variable], signed: bool = False)-> int:
    if isinstance(key, str):
        args = comm[key]
    elif isinstance(key, Variable):
        args = key
    elif isinstance(key, int):
        args = (key, 4)
    else:
        raise ValueError
    return int.from_bytes(target.read_memory_block8(*args), byteorder='little', signed=signed)


def disable_debug():
    """Disables the Debug block, reducing battery consumption."""
    target.write32(0x5C001004, 0x00000000)


def write_chunk_idx(idx: int) -> None:
    target.write32(comm["program_chunk_idx"].address, idx)


def write_chunk_count(count: int) -> None:
    target.write32(comm["program_chunk_count"].address, count)

def write_state(state: str) -> None:
    target.write32(comm["flashapp_state"].address, _flashapp_state_str_to_enum[state])


def extflash_erase(offset: int, size: int, whole_chip:bool = False, **kwargs) -> None:
    """Erase a range of data on extflash.

    On-device flashapp will round up to nearest minimum erase size.
    ``program_chunk_idx`` must externally be set.

    Parameters
    ----------
    offset: int
        Offset into extflash to erase.
    size: int
        Number of bytes to erase.
    whole_chip: bool
        If ``True``, ``size`` is ignored and the entire chip is erased.
        Defaults to ``False``.
    """
    global context_counter
    validate_extflash_offset(offset)
    if size <= 0 and not whole_chip:
        raise ValueError(f"Size must be >0; 0 erases the entire chip.")

    context = get_context()

    target.write32(context["address"].address, offset)
    target.write32(context["erase"].address, 1)       # Perform an erase at `program_address`
    target.write32(context["size"].address, 0)

    if whole_chip:
        target.write32(context["erase_bytes"].address, 0)    # Note: a 0 value erases the whole chip
    else:
        target.write32(context["erase_bytes"].address, size)

    target.write_memory_block8(context["expected_sha256"].address, _EMPTY_HASH_DIGEST)

    target.write32(context["ready"].address, context_counter)
    context_counter += 1

    wait_for_all_contexts_complete(**kwargs)


def extflash_read(offset: int, size: int) -> bytes:
    """Read data from extflash.

    Parameters
    ----------
    offset: int
        Offset into extflash to read.
    size: int
        Number of bytes to read.
    """
    validate_extflash_offset(offset)
    return bytes(target.read_memory_block8(0x9000_0000 + offset, size))


def extflash_write(offset:int,
                   data: bytes,
                   erase: bool = True,
                   blocking: bool = False,
                   decompressed_size: int = 0,
                   decompressed_hash: Optional[bytes]=None
                   ) -> None:
    """Write data to extflash.

    Limited to RAM constraints (i.e. <256KB writes).

    ``program_chunk_idx`` must externally be set.

    Parameters
    ----------
    offset: int
        Offset into extflash to write.
    size: int
        Number of bytes to write.
    erase: bool
        Erases flash prior to write.
        Defaults to ``True``.
    decompressed_size: int
        Size of decompressed data.
        0 if data has not been previously LZMA compressed.
    decompressed_hash: bytes
        SHA256 hash of the decompressed data
    """
    global context_counter
    validate_extflash_offset(offset)
    if not data:
        return
    if len(data) > (256 << 10):
        raise ValueError(f"Too large of data for a single write.")

    context = get_context()

    if blocking:
        wait_for("IDLE")
        target.halt()

    target.write32(context["address"].address, offset)
    target.write32(context["size"].address, len(data))

    if erase:
        target.write32(context["erase"].address, 1)       # Perform an erase at `program_address`

        if decompressed_size:
            target.write32(context["erase_bytes"].address, decompressed_size)
        else:
            target.write32(context["erase_bytes"].address, len(data))

    target.write32(context["decompressed_size"].address, decompressed_size)
    target.write_memory_block8(context["expected_sha256"].address, sha256(data))
    if decompressed_hash:
        target.write_memory_block8(context["expected_sha256_decompressed"].address, decompressed_hash)
    target.write_memory_block8(context["buffer"].address, data)

    target.write32(context["ready"].address, context_counter)
    context_counter += 1

    if blocking:
        target.resume()
        wait_for_all_contexts_complete()


def read_logbuf():
    return bytes(target.read_memory_block8(*comm["logbuf"])[:read_int("log_idx")]).decode()


def set_msp_pc(intflash_address):
    target.write_core_register('msp', read_int(intflash_address))
    target.write_core_register('pc', read_int(intflash_address + 4))


def start_flashapp(intflash_address):
    target.reset_and_halt()

    set_msp_pc(intflash_address)

    target.write32(comm["flashapp_state"].address, _flashapp_state_str_to_enum["INIT"])
    target.write32(comm["boot_magic"].address, 0xf1a5f1a5)  # Tell bootloader to boot into flashapp
    target.write32(comm["program_status"].address, 0)
    target.write32(comm["program_chunk_idx"].address, 1)  # Can be overwritten later
    target.write32(comm["program_chunk_count"].address, 100)  # Can be overwritten later
    target.resume()
    wait_for("IDLE")

    # Set game-and-watch RTC
    target.write32(comm["utc_timestamp"].address, timestamp_now())


def get_context(timeout=10):
    t_start = time()
    t_deadline = time() + timeout
    while True:
        for context in contexts:
            if not read_int(context["ready"]):
                return context
            if time() > t_deadline:
                raise TimeoutError

        sleep(sleep_duration)


def wait_for_all_contexts_complete(timeout=10):
    t_start = time()
    t_deadline = time() + timeout
    for context in contexts:
        while read_int(context["ready"]):
            if time() > t_deadline:
                raise TimeoutError
            sleep(sleep_duration)
    wait_for("IDLE", timeout = t_deadline - time())



def wait_for(status: str, timeout=10):
    """Block until the on-device status is matched."""
    t_start = time()
    t_deadline = time() + timeout
    error_mask = 0xFFFF_0000

    while True:
        status_enum = read_int("program_status")
        status_str = _flashapp_status_enum_to_str.get(status_enum, "UNKNOWN")
        if status_str == status:
            break
        elif (status_enum & error_mask) == 0xbad0_0000:
            raise DataError(status_str)
        if time() > t_deadline:
            raise TimeoutError
        sleep(sleep_duration)


def validate_extflash_offset(val):
    if val >= 0x9000_0000:
        raise ValueError(f"Provided extflash offset 0x{val:08X}, did you mean 0x{(val - 0x9000_0000):08X} ?")
    if val % 4096 != 0:
        raise ValueError(f"Extflash offset must be a multiple of 4096.")


################
# CLI Commands #
################
def flash(*, args, **kwargs):
    """Flash a binary to the external flash.

    Progress file format:
     * line 1: sha256 hash of extflash binary.
     * line 2: Number of chunks successfully flashed.
    """
    validate_extflash_offset(args.address)

    data = args.file.read_bytes()
    data_time = args.file.stat().st_mtime
    data_time = datetime.fromtimestamp(data_time).strftime('%Y-%m-%d %H:%M:%S:%f')

    chunk_size = contexts[0]["buffer"].size  # Assumes all contexts have same size buffer
    chunks = chunk_bytes(data, chunk_size)
    total_n_chunks = len(chunks)

    previous_chunks_already_flashed = 0
    # Attempt to resume a previous session.
    if args.progress_file and args.progress_file.exists():
        progress_file_time, progress_file_chunks_already_flashed = args.progress_file.read_text().split("\n")
        progress_file_chunks_already_flashed = int(progress_file_chunks_already_flashed)
        if progress_file_time == data_time:
            previous_chunks_already_flashed = progress_file_chunks_already_flashed
            print(f"Resuming previous session at {previous_chunks_already_flashed}/{total_n_chunks}")

    # https://github.com/tqdm/tqdm/issues/1264
    chunks = chunks[previous_chunks_already_flashed:]
    compressed_chunks = compress_chunks(chunks)

    write_chunk_count(total_n_chunks);

    base_address = args.address + (previous_chunks_already_flashed * chunk_size)
    with tqdm(initial=previous_chunks_already_flashed, total=total_n_chunks) as pbar:
        for i, (chunk, compressed_chunk) in enumerate(zip(chunks, compress_chunks(chunks))):
            chunk_1_idx = previous_chunks_already_flashed + i + 1
            pbar.update(1)
            if len(compressed_chunk) < len(chunk):
                decompressed_size = len(chunk)
                decompressed_hash = sha256(chunk)
                chunk = compressed_chunk
            else:
                decompressed_size = 0
                decompressed_hash = None
            write_chunk_idx(chunk_1_idx)
            extflash_write(base_address + (i * chunk_size), chunk,
                           decompressed_size=decompressed_size,
                           decompressed_hash=decompressed_hash,
                           )

            # Save current progress to a file incase progress is interrupted.
            if args.progress_file:
                # Up to 3 chunks may have been sent to device that may have NOT been written to disk.
                # This is the most conservative estimate of what has been written to disk.
                chunks_already_flashed = max(
                    previous_chunks_already_flashed,
                    chunk_1_idx - 3
                )
                args.progress_file.parent.mkdir(exist_ok=True, parents=True)
                args.progress_file.write_text(f"{data_time}\n{chunks_already_flashed}")

        wait_for_all_contexts_complete()
        wait_for("IDLE")

    if args.progress_file and args.progress_file.exists():
        args.progress_file.unlink()


def erase(**kwargs):
    """Erase the entire external flash."""
    # Just setting an artibrarily long timeout
    extflash_erase(0, 0, whole_chip=True, timeout=10_000)


def _ls(fs, path):
    try:
        for element in fs.scandir(path):
            if element.type == 1:
                typ = "FILE"
            elif element.type == 2:
                typ = "DIR "
            else:
                typ = "UKWN"

            fullpath = f"{path}/{element.name}"
            try:
                time_val = int.from_bytes(fs.getattr(fullpath, "t"), byteorder='little')
                time_str = datetime.fromtimestamp(time_val, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            except LittleFSError:
                time_str = " " * 19

            print(f"{element.size:7}B {typ} {time_str} {element.name}")
    except LittleFSError as e:
        if e.code != -2:
            raise
        print(f"ls {path}: No such directory")


def ls(*, args, fs, **kwargs):
    _ls(fs, args.path.as_posix())


def pull(*, args, fs, **kwargs):
    try:
        stat = fs.stat(args.gnw_path.as_posix())
    except LittleFSError as e:
        if e.code != -2:
            raise
        print(f"{args.gnw_path.as_posix()}: No such file or directory")
        return

    if stat.type == 1:  # file
        with fs.open(args.gnw_path.as_posix(), 'rb') as f:
            data = f.read()
        if args.local_path.is_dir():
            args.local_path = args.local_path / args.gnw_path.name
        args.local_path.write_bytes(data)
    elif stat.type == 2:  # dir
        if args.local_path.is_file():
            raise ValueError(f"Cannot backup directory \"{args.gnw_path.as_posix()}\" to file \"{args.local_path}\"")

        strip_root = not args.local_path.exists()
        for root, _, files in fs.walk(args.gnw_path.as_posix()):
            root = Path(root.lstrip("/"))
            for file in files:
                full_src_path = root / file

                if strip_root:
                    full_dst_path = args.local_path / Path(*full_src_path.parts[1:])
                else:
                    full_dst_path = args.local_path / full_src_path

                full_dst_path.parent.mkdir(exist_ok=True, parents=True)

                if args.verbose:
                    print(f"{full_src_path}  ->  {full_dst_path}")

                with fs.open(full_src_path.as_posix(), 'rb') as f:
                    data = f.read()

                full_dst_path.write_bytes(data)
    else:
        raise NotImplementedError(f"Unknown type: {stat.type}")


def push(*, args, fs, **kwargs):
    gnw_path_is_dir = is_existing_gnw_dir(fs, args.gnw_path)

    for local_path in args.local_paths:
        if not local_path.exists():
            raise ValueError(f"Local \"{local_path}\" does not exist.")

        if local_path.is_file():
            data = local_path.read_bytes()

            if gnw_path_is_dir:
                gnw_path = args.gnw_path / local_path.name
            else:
                gnw_path = args.gnw_path

            if args.verbose:
                print(f"{local_path}  ->  {gnw_path.as_posix()}")

            if sha256(data) != gnw_sha256(fs, gnw_path):
                fs.makedirs(gnw_path.parent.as_posix(), exist_ok=True)

                with fs.open(gnw_path.as_posix(), "wb") as f:
                    f.write(data)

            fs.setattr(gnw_path.as_posix(), 't', timestamp_now_bytes())
        else:
            for file in local_path.rglob("*"):
                if file.is_dir():
                    continue
                data = file.read_bytes()

                subpath = file.relative_to(local_path.parent)
                if not gnw_path_is_dir:
                    gnw_path = args.gnw_path / Path(*subpath.parts[1:])
                else:
                    gnw_path = args.gnw_path / subpath

                if args.verbose:
                    print(f"{file}  ->  {gnw_path.as_posix()}")

                if sha256(data) != gnw_sha256(fs, gnw_path):
                    fs.makedirs(gnw_path.parent.as_posix(), exist_ok=True)

                    with fs.open(gnw_path.as_posix(), "wb") as f:
                        f.write(data)

                fs.setattr(gnw_path.as_posix(), 't', timestamp_now_bytes())
    wait_for_all_contexts_complete()


def format(*, args, fs, **kwargs):
    # TODO: add a confirmation prompt and a --force option
    fs.format()


def rm(*, args, fs, **kwargs):
    try:
        stat = fs.stat(args.path.as_posix())
    except LittleFSError as e:
        if e.code != -2:
            raise
        print(f"{args.path.as_posix()}: No such file or directory")
        return

    if stat.type == 1:  # file
        fs.remove(args.path.as_posix())
    elif stat.type == 2:  # dir
        for root, dirs, files in reversed(list(fs.walk(args.path.as_posix()))):
            root = Path(root)
            for typ in (files, dirs):
                for name in typ:
                    full_path = root / name
                    fs.remove(full_path.as_posix())
        fs.remove(args.path.as_posix())
    else:
        raise NotImplementedError(f"Unknown type: {stat.type}")


def mkdir(*, args, fs, **kwargs):
    fs.makedirs(args.path.as_posix(), exist_ok=True)

def mv(*, args, fs, **kwargs):
    fs.rename(args.src.as_posix(), args.dst.as_posix())


def shell(*, args, parser, **kwargs):
    print("Interactive shell. Press Ctrl-D to exit.")

    while True:
        try:
            user_input = input("gnw$ ")
        except EOFError:
            return
        if not user_input:
            continue

        split_user_input = shlex.split(user_input)

        command = split_user_input[0]
        if command == "help":
            parser.print_help()
            continue
        if command not in commands:
            print(f"Invalid command: {split_user_input[0]}")
            continue
        if "--help" in split_user_input:
            subparsers.choices[command].print_help()
            continue

        try:
            parsed = parser.parse_args(split_user_input, copy(args))
        except SystemExit as e:
            continue

        if parsed.command == "shell":
            print("Cannot nest shells.")
            continue

        try:
            commands[parsed.command](args=parsed, parser=parser, **kwargs)
        except Exception as e:
            print(e)
            continue


def screenshot(*, args, fs, **kwargs):
    try:
        with fs.open("SCREENSHOT", "rb") as f:
            tamp_compressed_data = f.read()
    except FileNotFoundError:
        print("No screenshot found on device.")
        sys.exit(1)
    data = tamp.decompress(tamp_compressed_data)
    # Convert raw RGB565 pixel data to PNG
    img = Image.new("RGB", (320, 240))
    pixels = img.load()
    index = 0
    for y in range(240):
        for x in range(320):
            color, = struct.unpack('<H', data[index:index+2])
            red =   int(((color & 0b1111100000000000) >> 11) / 31.0 * 255.0)
            green = int(((color & 0b0000011111100000) >>  5) / 63.0 * 255.0)
            blue =  int(((color & 0b0000000000011111)      ) / 31.0 * 255.0)
            pixels[x, y] = (red, green, blue)
            index += 2

    img.save(args.output)


def main():
    global commands, subparsers
    commands = {}

    global_parser = argparse.ArgumentParser(add_help=False)
    global_parser.add_argument('--verbose', action='store_true')
    global_parser.add_argument("--no-disable-debug", action="store_true",
                        help="Don't disable the debug hw block after flashing.")
    group = global_parser.add_mutually_exclusive_group()
    group.add_argument("--intflash-bank", type=int, default=1, choices=(1, 2),
                        help="Retro Go internal flash bank")
    valid_intflash_bank_1_addresses = set(range(0x0800_0000, (0x0800_0000 + (256 << 10)), 4))
    valid_intflash_bank_2_addresses = set(range(0x0810_0000, (0x0810_0000 + (256 << 10)), 4))
    group.add_argument("--intflash-address",
                       type=lambda x: int(x,0),
                       help="Retro Go internal flash address.")

    parser = argparse.ArgumentParser(
        prog="gnwmanager",
        parents=[global_parser],
        description="Multiple commands may be given in a single session, delimited be \"--\"",
    )
    subparsers = parser.add_subparsers(dest="command")


    def add_command(handler):
        """Add a subcommand, like "flash"."""
        subparser = subparsers.add_parser(handler.__name__, parents=[global_parser])
        commands[handler.__name__] = handler
        return subparser

    subparser = add_command(flash)
    subparser.add_argument("file", type=Path,
            help="Binary file to flash.")
    subparser.add_argument("address", type=lambda x: int(x,0),
            help="Offset into external flash")
    subparser.add_argument("--progress-file", type=Path,
            help="Save/Load progress from a file; allows resuming a interrupted flash operation.")

    subparser = add_command(erase)

    subparser = add_command(ls)
    subparser.add_argument('path', nargs='?', type=Path, default='',
            help="Directory to list the contents of. Defaults to root.")

    subparser = add_command(pull)
    subparser.add_argument("gnw_path", type=Path,
            help="Game-and-watch file or folder to copy to computer.")
    subparser.add_argument("local_path", type=Path,
            help="Local file or folder to copy data to.")

    subparser = add_command(push)
    subparser.add_argument("gnw_path", type=Path,
            help="Game-and-watch file or folder to write to.")
    subparser.add_argument("local_paths", nargs='+', type=Path,
            help="Local file or folder to copy data from.")

    subparser = add_command(rm)
    subparser.add_argument('path', type=Path,
            help="File or folder to delete.")

    subparser = add_command(mkdir)
    subparser.add_argument('path', type=Path,
            help="Directory to create.")

    subparser = add_command(mv)
    subparser.add_argument('src', type=Path,
            help="Source file or directory.")
    subparser.add_argument('dst', type=Path,
            help="Destination file or directory.")

    subparser = add_command(format)

    subparser = add_command(screenshot)
    subparser.add_argument('output', nargs='?', type=Path, default=Path("screenshot.png"),
            help="Destination file or directory.")

    subparser = add_command(shell)

    parser.set_defaults(command='shell')

    # Separate commands and their arguments based on '--'
    sys_args = sys.argv[1:]
    global_args = []
    for i, arg in enumerate(sys_args):
        if arg in commands:
            sys_args = sys_args[i:]
            break
        else:
            global_args.append(arg)

    commands_args = []
    current_command_args = []
    for arg in sys_args:
        if arg == '--':
            commands_args.append(current_command_args)
            current_command_args = []
        else:
            current_command_args.append(arg)
    commands_args.append(current_command_args)

    parsed_args = []
    for command_args in commands_args:
        current_args = parser.parse_args(command_args + global_args)

        if current_args.intflash_address is None:
            current_args.intflash_address = 0x0800_0000 if current_args.intflash_bank == 1 else 0x0810_0000

        if current_args.intflash_address in valid_intflash_bank_1_addresses:
            current_args.intflash_bank = 1
        elif current_args.intflash_address in valid_intflash_bank_2_addresses:
            current_args.intflash_bank = 2
        else:
            raise NotImplementedError

        parsed_args.append(current_args)

    options = {
        "frequency": 5_000_000,
    }

    session = ConnectHelper.session_with_chosen_probe(options=options)
    session.board.target = STM32H7B0_256K(session)

    try:
        with session:
            global target
            board = session.board
            assert board is not None
            target = board.target

            if False:
                # This successfully erases
                eraser = FlashEraser(session, FlashEraser.Mode.SECTOR)
                eraser.erase([
                    (0x0800_0000, 0x0800_0000 + (256 * 1024)),
                    (0x0810_0000, 0x0810_0000 + (256 * 1024)),
                ])
            programmer = FileProgrammer(session, progress=None, no_reset=False)
            programmer.program("build/gw_retro_go_intflash.bin", base_address=0x0800_0000)
            #programmer.program("build/gw_retro_go_intflash.bin", base_address=0x0810_0000)
            target.reset()
            breakpoint()

            try:
                for i, args in enumerate(parsed_args):
                    if args.intflash_address is None:
                        args.intflash_address = 0x0800_0000 if args.intflash_bank == 1 else 0x0810_0000

                    if args.intflash_address in valid_intflash_bank_1_addresses:
                        args.intflash_bank = 1
                    elif args.intflash_address in valid_intflash_bank_2_addresses:
                        args.intflash_bank = 2
                    else:
                        raise NotImplementedError

                    if i == 0:
                        start_flashapp(args.intflash_address)

                        filesystem_offset = read_int("lfs_cfg_context") - 0x9000_0000
                        block_size = read_int("lfs_cfg_block_size")
                        block_count = read_int("lfs_cfg_block_count")

                        if block_size==0 or block_count==0:
                            raise DataError

                        lfs_context = LfsDriverContext(filesystem_offset)
                        fs = LittleFS(lfs_context, block_size=block_size, block_count=block_count)

                    commands[args.command](
                        args=args,
                        fs=fs,
                        block_size=block_size,
                        block_count=block_count,
                        parser=parser,
                    )
            finally:
                if not args.no_disable_debug:
                    disable_debug()

                target.reset_and_halt()
                set_msp_pc(args.intflash_address)
                target.resume()
    except usb.core.USBError as e:
        new_message = str(e) + "\n\n\nTry unplugging and replugging in your adapter.\n\n"
        raise type(e)(new_message) from e
    except ProbeError as e:
        new_message = str(e) + "\n\n\nIs your Game & Watch on?\n\n"
        raise type(e)(new_message) from e


if __name__ == "__main__":
    main()
