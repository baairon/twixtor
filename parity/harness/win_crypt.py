"""The Windows CryptoAPI context functions: CryptAcquireContextA/W, CryptReleaseContext, CryptGenRandom.

Return values and GetLastError values follow the documented behaviour of the Microsoft providers for
the cases a process with no key store can reach. Containers live only in this process: nothing is read
from the host, no machine or user identity is used, and no key is ever created.

Stand-in: CryptGenRandom returns a deterministic byte stream (SHA-256 in counter mode over a fixed seed,
one stream shared by all contexts), not random bytes. Only names, sizes, flags and return values are
logged, never buffers or strings.
"""
import hashlib
import struct

TRUE, FALSE = 1, 0

ERROR_INVALID_PARAMETER = 87
ERROR_NOACCESS = 998
NTE_BAD_UID = 0x80090001
NTE_BAD_FLAGS = 0x80090009
NTE_EXISTS = 0x8009000F
NTE_BAD_PROV_TYPE = 0x80090014
NTE_BAD_KEYSET = 0x80090016
NTE_PROV_TYPE_NOT_DEF = 0x80090017
NTE_KEYSET_NOT_DEF = 0x80090019
NTE_PROV_TYPE_NO_MATCH = 0x8009001B

CRYPT_NEWKEYSET = 0x8
CRYPT_DELETEKEYSET = 0x10
CRYPT_MACHINE_KEYSET = 0x20
CRYPT_SILENT = 0x40
CRYPT_DEFAULT_CONTAINER_OPTIONAL = 0x80
CRYPT_VERIFYCONTEXT = 0xF0000000
KNOWN_FLAGS = (CRYPT_NEWKEYSET | CRYPT_DELETEKEYSET | CRYPT_MACHINE_KEYSET | CRYPT_SILENT
               | CRYPT_DEFAULT_CONTAINER_OPTIONAL | CRYPT_VERIFYCONTEXT)

SEED = b"twixtor-parity-harness/CryptGenRandom/v1"

# provider name -> provider type
PROVIDERS = {
    "microsoft base cryptographic provider v1.0": 1,
    "microsoft enhanced cryptographic provider v1.0": 1,
    "microsoft strong cryptographic provider": 1,
    "microsoft base dss cryptographic provider": 3,
    "microsoft base dss and diffie-hellman cryptographic provider": 13,
    "microsoft dh schannel cryptographic provider": 18,
    "microsoft rsa schannel cryptographic provider": 12,
    "microsoft enhanced rsa and aes cryptographic provider": 24,
    "microsoft enhanced rsa and aes cryptographic provider (prototype)": 24,
}
# provider types that have a default provider
DEFAULT_TYPES = {1, 3, 12, 13, 18, 24}


class Stream:
    """SHA-256(seed || counter) blocks, consumed in order."""

    def __init__(self, seed=SEED):
        self.seed, self.counter, self.buf = seed, 0, b""

    def read(self, n):
        while len(self.buf) < n:
            self.buf += hashlib.sha256(self.seed + struct.pack("<Q", self.counter)).digest()
            self.counter += 1
        out, self.buf = self.buf[:n], self.buf[n:]
        return out


class Crypt:
    def __init__(self):
        self.contexts = {}          # handle -> (provider type, ephemeral)
        self.containers = set()     # (name or None, machine keyset)
        self.next_handle = 0x1000
        self.stream = Stream()


def read_cstr(ql, addr, wide, limit=512):
    out = bytearray()
    step = 2 if wide else 1
    while len(out) < limit:
        c = bytes(ql.mem.read(addr + len(out), step))
        if c == bytes(step):
            break
        out += c
    return out.decode("utf-16le" if wide else "latin-1", "replace")


def install(stubs, arg, ret, fault_type=None):
    state = Crypt()

    def fail(emu, ql, name, code, **kw):
        ql.os.last_error = code
        emu.rec.log("crypt", api=name, ret=0, last_error=hex(code), **kw)
        ret(ql, FALSE)

    def acquire(wide, name):
        def handler(emu, ql):
            prov_p, cont_p, name_p, ptype, flags = (arg(ql, i) for i in range(5))
            ptype, flags = ptype & 0xFFFFFFFF, flags & 0xFFFFFFFF
            kw = dict(prov_type=ptype, flags=hex(flags), container_given=bool(cont_p), provider_given=bool(name_p))
            if not prov_p:
                return fail(emu, ql, name, ERROR_INVALID_PARAMETER, **kw)
            if flags & ~KNOWN_FLAGS & 0xFFFFFFFF or (flags & CRYPT_VERIFYCONTEXT) not in (0, CRYPT_VERIFYCONTEXT):
                return fail(emu, ql, name, NTE_BAD_FLAGS, **kw)
            verify = (flags & CRYPT_VERIFYCONTEXT) == CRYPT_VERIFYCONTEXT
            if verify and (flags & ~CRYPT_VERIFYCONTEXT & ~CRYPT_SILENT & ~CRYPT_MACHINE_KEYSET & 0xFFFFFFFF or cont_p):
                return fail(emu, ql, name, NTE_BAD_FLAGS, **kw)
            if ptype < 1 or ptype > 999:
                return fail(emu, ql, name, NTE_BAD_PROV_TYPE, **kw)
            if name_p:
                found = PROVIDERS.get(read_cstr(ql, name_p, wide).lower())
                if found is None:
                    return fail(emu, ql, name, NTE_KEYSET_NOT_DEF, **kw)
                if found != ptype:
                    return fail(emu, ql, name, NTE_PROV_TYPE_NO_MATCH, **kw)
            elif ptype not in DEFAULT_TYPES:
                return fail(emu, ql, name, NTE_PROV_TYPE_NOT_DEF, **kw)
            key = (read_cstr(ql, cont_p, wide) if cont_p else None, bool(flags & CRYPT_MACHINE_KEYSET))
            if not verify:
                if flags & CRYPT_DELETEKEYSET:
                    if key not in state.containers:
                        return fail(emu, ql, name, NTE_BAD_KEYSET, **kw)
                    state.containers.discard(key)
                    emu.rec.log("crypt", api=name, ret=1, deleted_keyset=True, **kw)
                    return ret(ql, TRUE)
                if flags & CRYPT_NEWKEYSET:
                    if key in state.containers:
                        return fail(emu, ql, name, NTE_EXISTS, **kw)
                    state.containers.add(key)
                elif key not in state.containers:
                    return fail(emu, ql, name, NTE_BAD_KEYSET, **kw)
            state.next_handle += 0x10
            state.contexts[state.next_handle] = ptype
            ql.mem.write(prov_p, struct.pack("<Q", state.next_handle))
            emu.rec.log("crypt", api=name, ret=1, **kw)
            ret(ql, TRUE)
        return handler

    def release(emu, ql):
        handle, flags = arg(ql, 0), arg(ql, 1) & 0xFFFFFFFF
        if flags:
            return fail(emu, ql, "CryptReleaseContext", NTE_BAD_FLAGS, flags=hex(flags))
        if handle not in state.contexts:
            return fail(emu, ql, "CryptReleaseContext", NTE_BAD_UID, flags=hex(flags))
        del state.contexts[handle]
        emu.rec.log("crypt", api="CryptReleaseContext", ret=1, flags=hex(flags))
        ret(ql, TRUE)

    def gen_random(emu, ql):
        handle, length, buf = arg(ql, 0), arg(ql, 1) & 0xFFFFFFFF, arg(ql, 2)
        if handle not in state.contexts:
            return fail(emu, ql, "CryptGenRandom", NTE_BAD_UID, size=length)
        if length == 0:
            emu.rec.log("crypt", api="CryptGenRandom", ret=1, size=0)
            return ret(ql, TRUE)
        if not buf:
            return fail(emu, ql, "CryptGenRandom", ERROR_NOACCESS, size=length)
        ql.mem.write(buf, state.stream.read(length))
        emu.rec.log("crypt", api="CryptGenRandom", ret=1, size=length, stand_in="deterministic byte stream")
        ret(ql, TRUE)

    stubs["CryptAcquireContextA"] = acquire(False, "CryptAcquireContextA")
    stubs["CryptAcquireContextW"] = acquire(True, "CryptAcquireContextW")
    stubs["CryptReleaseContext"] = release
    stubs["CryptGenRandom"] = gen_random
    return state
