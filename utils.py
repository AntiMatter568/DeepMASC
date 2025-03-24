import asyncio
import os


async def run_subprocess(cmd):
    # Create the subprocess with pipes for stdout and stderr
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=dict(os.environ, PYTHONUNBUFFERED="1"),
    )

    # Asynchronous function to read and print lines from a stream
    async def read_stream(stream, is_stderr):
        while True:
            line = await stream.readline()
            if not line:
                break
            line_stripped = line.decode().strip()
            if line_stripped:
                if is_stderr and ("error" in line_stripped.lower() or "exception" in line_stripped.lower()):
                    print(f"ERROR: {line_stripped}", flush=True)
                else:
                    print(line_stripped, flush=True)

    # Run both stream readers concurrently
    await asyncio.gather(
        read_stream(proc.stdout, False),
        read_stream(proc.stderr, True),
    )

    # Wait for the subprocess to finish
    await proc.wait()
    return proc.returncode
