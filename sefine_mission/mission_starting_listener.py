import asyncio
from mavsdk import System
import subprocess
from pool_test import main

mission_process = None


async def listen_states(drone : System):

    flag = False

    async for status in drone.telemetry.status_text():

        message = status.text.strip()
        print(f"The coming message: {message}")

        if flag == False and message == "START_MISSION":
            print(f"The mission is starting...")
            flag = True
            # mission_process = subprocess.Popen(["python3", "pool_test.py"])
            main(drone=drone)
            return
        


async def run():

    drone = System()

    await drone.connect(system_address="serial://ttyUsb:115200") # It will be indicated later.

    print("Connecting...")

    starting_time = asyncio.get_event_loop().time()

    async for state in drone.core.connection_state():

        if state.is_connected:
            print("The vehicle has been connected successfully.")
            break

        passed_time = asyncio.get_event_loop().time() - starting_time

        if passed_time >= 10:
            print("Timeout. Control device ccould not be connected.")
            print("It is being closed...")
            exit()
        
    listen_states(drone)


if __name__ == "__main__":

    asyncio.run(run)