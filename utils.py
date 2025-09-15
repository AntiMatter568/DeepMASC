import asyncio
import os
import subprocess


async def run_subprocess(cmd, env=None, cwd=None):
    # Create the subprocess with pipes for stdout and stderr
    # Set up environment variables
    subprocess_env = dict(os.environ, PYTHONUNBUFFERED="1")
    if env:
        subprocess_env.update(env)
    
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=subprocess_env,
        cwd=cwd,
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


def run_subprocess_realtime(cmd, timeout=None):
    """
    Run subprocess with real-time output without using asyncio.
    This avoids event loop conflicts while still providing live output.
    """
    import time
    
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout
        universal_newlines=True,
        bufsize=1,  # Line buffered
        env=env
    )
    
    start_time = time.time()
    
    # Read output line by line in real-time
    while True:
        # Check timeout
        if timeout and (time.time() - start_time) > timeout:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise subprocess.TimeoutExpired(cmd, timeout)
        
        # Read line with timeout
        line = proc.stdout.readline()
        if line:
            print(line.strip(), flush=True)
        elif proc.poll() is not None:
            # Process has finished
            break
        else:
            # No output yet, wait a bit
            time.sleep(0.1)
    
    # Get any remaining output
    remaining_output = proc.stdout.read()
    if remaining_output:
        print(remaining_output.strip(), flush=True)
    
    return proc.returncode