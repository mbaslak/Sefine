from mavsdk import System
from mavsdk.offboard import VelocityBodyYawspeed
import asyncio 



await drone.offboard.set_velocity_body(VelocityBodyYaw(0.0, 0.0, 0.0, 36.0))

asyncio.sleep(10)

await drone.offboard.set_velocity_body(VelocityBodyYaw(0.0, 0.0, 0.0, 0.0))