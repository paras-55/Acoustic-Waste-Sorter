"""
esp_link.py - the ONE place that talks to the ESP32 (WasteSorter firmware).

Used by: 4_Integration/mic_check.py, 2_Audio_Dataset/record_audio.py,
         4_Integration/sorter_controller.py

Protocol (see WasteSorter.ino): text lines end in '\n'; audio arrives only as
a framed block  A5 5A 'W' 'A' 'V' | status | slot | nSamples(u32) | int16[] | byteSum(u32)
"""
import struct
import time

import numpy as np

PORT = "COM5"
BAUD = 921600
MAGIC = b"\xA5\x5AWAV"
SAMPLE_RATE = 16000

STATUS_IMMEDIATE, STATUS_TRIGGERED, STATUS_TIMEOUT = 0, 1, 2


class EspError(RuntimeError):
    pass


class EspLink:
    def __init__(self, port=PORT, baud=BAUD, verbose=True, serial_obj=None):
        self.verbose = verbose
        self.boot_lines = []
        if serial_obj is not None:          # used by the offline tests
            self.ser = serial_obj
        else:
            import serial                    # pyserial
            try:
                self.ser = serial.Serial(port, baud, timeout=0.5)
            except serial.SerialException as e:
                raise EspError(
                    f"Cannot open {port}: {e}\n"
                    "  -> Close the Arduino IDE Serial Monitor / Serial Plotter "
                    "(only one program can use the port) and check the COM number."
                ) from None
        self._wait_ready()

    # ---------------- low level ----------------
    def _log(self, msg):
        if self.verbose:
            print(f"  [ESP32] {msg}")

    def _read_exact(self, n, timeout):
        buf = bytearray()
        end = time.time() + timeout
        while len(buf) < n:
            chunk = self.ser.read(n - len(buf))
            if chunk:
                buf += chunk
            elif time.time() > end:
                raise EspError(f"Timed out: got {len(buf)} of {n} bytes")
        return bytes(buf)

    def read_line(self, timeout=2.0):
        """Return one text line (without newline) or None on timeout."""
        buf = bytearray()
        end = time.time() + timeout
        while time.time() < end:
            b = self.ser.read(1)
            if not b:
                continue
            if b == b"\n":
                return buf.decode(errors="replace").strip()
            buf += b
        return None

    def _wait_ready(self, timeout=6.0):
        # Opening the port resets most DevKit boards; the firmware prints its
        # diagnostic and then READY. If the board did not reset, ask with '?'.
        start = time.time()
        asked = False
        while time.time() - start < timeout:
            line = self.read_line(timeout=1.0)
            if line is None:
                if not asked and time.time() - start > 2.5:
                    self.ser.write(b"?")
                    asked = True
                continue
            if line:
                self.boot_lines.append(line)
                self._log(line)
            if line == "READY" or (asked and line.startswith("STATUS")):
                return
        raise EspError(
            "ESP32 did not answer. Is WasteSorter.ino uploaded, is the port COM5, "
            "and is the Serial Monitor closed?"
        )

    def flush_input(self):
        time.sleep(0.05)
        self.ser.reset_input_buffer()

    def send(self, text):
        self.ser.write(text.encode())

    def command(self, text, expect_prefix, timeout=4.0):
        """Send a text command and wait for a reply line starting with expect_prefix."""
        self.flush_input()
        self.send(text)
        end = time.time() + timeout
        while time.time() < end:
            line = self.read_line(timeout=end - time.time())
            if line is None:
                break
            if line:
                self._log(line)
            if line.startswith(expect_prefix):
                return line
        raise EspError(f"No '{expect_prefix}' reply to command {text!r}")

    # ---------------- high level ----------------
    def diagnostic(self):
        """Run 'D'; return dict with lines, ok flag and chosen slot."""
        self.flush_input()
        self.send("D")
        lines, end = [], time.time() + 8
        while time.time() < end:
            line = self.read_line(timeout=end - time.time())
            if line is None:
                break
            if line:
                lines.append(line)
                self._log(line)
            if line.startswith("STATUS"):
                break
        result = next((l for l in lines if l.startswith("DIAG RESULT")), "")
        slot = None
        if "slot=" in result:
            try:
                slot = int(result.split("slot=")[1].split()[0])
            except ValueError:
                pass
        return {"lines": lines, "ok": "MIC_OK" in result, "slot": slot}

    def set_gain_shift(self, n):
        return self.command(f"G{int(n)}\n", "GAIN_OK")

    def record(self, triggered=True, timeout=25.0):
        """
        Request a clip. Returns (samples_int16 ndarray, status) or (None, STATUS_TIMEOUT)
        when the trigger timed out. Raises EspError on a corrupted transfer.
        """
        self.flush_input()
        self.send("T" if triggered else "R")
        # Read until the magic header, echoing any text lines.
        window = bytearray()
        text = bytearray()
        end = time.time() + timeout
        while True:
            b = self.ser.read(1)
            if not b:
                if time.time() > end:
                    raise EspError("Timed out waiting for audio frame header")
                continue
            window += b
            if len(window) > len(MAGIC):
                del window[0]
            if bytes(window) == MAGIC:
                break
            if b == b"\n":
                line = text.decode(errors="replace").strip()
                text.clear()
                if line:
                    self._log(line)
            else:
                text += b
        status, slot = struct.unpack("<BB", self._read_exact(2, 3))
        (n,) = struct.unpack("<I", self._read_exact(4, 3))
        if n > 10 * SAMPLE_RATE:
            raise EspError(f"Malformed frame: nSamples={n}")
        payload = self._read_exact(2 * n, 10) if n else b""
        (checksum,) = struct.unpack("<I", self._read_exact(4, 3))
        if (sum(payload) & 0xFFFFFFFF) != checksum:
            raise EspError("Checksum mismatch: audio corrupted in transfer (re-record)")
        if status == STATUS_TIMEOUT:
            return None, status
        return np.frombuffer(payload, dtype="<i2").copy(), status

    def sort(self, label):
        """label: 'plastic' | 'metal' | 'center'. Blocks until the tray is back."""
        cmd = {"plastic": ("P", "PLASTIC_OK"), "metal": ("M", "METAL_OK"),
               "center": ("C", "CENTER_OK")}[label]
        return self.command(cmd[0], cmd[1], timeout=5.0)

    def angle(self, deg):
        return self.command(f"A{int(deg)}\n", "ANGLE_OK")

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
