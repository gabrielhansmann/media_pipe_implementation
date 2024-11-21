import pyrealsense2 as rs

def stringify_device_list():
    context = rs.context()
    devices = context.query_devices()
    devices = [f"{device.get_info(rs.camera_info.name)} (SN: {device.get_info(rs.camera_info.serial_number)})" for device in devices]
    return devices
