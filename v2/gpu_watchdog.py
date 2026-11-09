"""Pause the training process while the GPU is hot, and continue it once cooled down.

The process is suspended, not killed, so no epoch is lost. Log: gpu_watchdog.log.
    python gpu_watchdog.py [--stop 82] [--resume 68] [--selftest]
It exits (after resuming the target) when no training process has existed for 5 minutes.
"""
import argparse, ctypes, json, subprocess, time
from pathlib import Path

TARGET = "xray_finetuning_v2"
LOG = Path(__file__).resolve().parent / "gpu_watchdog.log"
ntdll, kernel32 = ctypes.WinDLL("ntdll"), ctypes.WinDLL("kernel32", use_last_error=True)
kernel32.OpenProcess.restype = ctypes.c_void_p
ntdll.NtSuspendProcess.argtypes = ntdll.NtResumeProcess.argtypes = [ctypes.c_void_p]
kernel32.CloseHandle.argtypes = [ctypes.c_void_p]


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def gpu():
    out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,power.draw", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=20).stdout.strip()
    t, u, p = [x.strip() for x in out.split(",")]
    return int(t), u, p


def targets():
    ps = (f"Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object {{ $_.CommandLine -like '*{TARGET}*' }} "
          "| Select-Object -ExpandProperty ProcessId | ConvertTo-Json -Compress")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=60).stdout.strip()
    if not out:
        return []
    v = json.loads(out)
    return v if isinstance(v, list) else [v]


def act(pids, fn):
    for pid in pids:
        h = kernel32.OpenProcess(0x0800, False, pid)   # PROCESS_SUSPEND_RESUME
        if not h:
            log(f"cannot open process {pid}: error {ctypes.get_last_error()}")
            continue
        status = fn(h)
        kernel32.CloseHandle(h)
        if status:
            log(f"NT status {status:#x} for process {pid}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop", type=int, default=82)
    ap.add_argument("--resume", type=int, default=68)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

    if a.selftest:
        pids = targets()
        log(f"selftest pids={pids} gpu={gpu()}")
        act(pids, ntdll.NtSuspendProcess); time.sleep(25)
        log(f"selftest while paused gpu={gpu()}")
        act(pids, ntdll.NtResumeProcess); time.sleep(25)
        log(f"selftest after resume gpu={gpu()}")
        return

    log(f"watchdog on: pause at {a.stop} C or above, continue at {a.resume} C or below")
    paused, paused_at, peak, last_report, idle_since, n_pauses, failures = [], 0.0, 0, 0.0, None, 0, 0
    try:
        while True:
            try:
                t, u, p = gpu()
                failures = 0
            except Exception as e:
                failures += 1
                log(f"temperature read failed ({failures}): {type(e).__name__}")
                if failures >= 4 and not paused:   # blind to the temperature, so stay on the safe side
                    paused = targets(); act(paused, ntdll.NtSuspendProcess); paused_at = time.time()
                    log("paused because the temperature cannot be read")
                time.sleep(30)
                continue
            peak = max(peak, t)
            pids = paused or targets()
            if not pids:
                idle_since = idle_since or time.time()
                if time.time() - idle_since > 300:
                    log(f"no training process for 5 minutes, watchdog off. peak {peak} C, pauses {n_pauses}")
                    return
            else:
                idle_since = None
            if not paused and pids and t >= a.stop:
                act(pids, ntdll.NtSuspendProcess)
                paused, paused_at = pids, time.time()
                n_pauses += 1
                log(f"PAUSED at {t} C (pause #{n_pauses})")
            elif paused and t <= a.resume:
                act(paused, ntdll.NtResumeProcess)
                log(f"RESUMED at {t} C after {(time.time() - paused_at) / 60:.1f} min")
                paused = []
            if time.time() - last_report > 600:
                log(f"status {t} C, util {u}%, power {p} W, peak {peak} C, {'paused' if paused else 'running'}")
                last_report = time.time()
            time.sleep(10)
    finally:
        if paused:   # never leave the training frozen behind a dead watchdog
            act(paused, ntdll.NtResumeProcess)
            log("watchdog ending: training resumed")


if __name__ == "__main__":
    main()
