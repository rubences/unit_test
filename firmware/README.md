# Firmware

This directory contains C++ firmware code for deployment on Edge AI microcontrollers (ESP32, Arduino Nicla Sense).

## Structure

- `src/` - Main firmware source code
- `include/` - Header files
- `lib/` - External libraries (BNO055 driver, TensorFlow Lite Micro)
- `platformio.ini` - PlatformIO configuration

## Hardware Support

- **ESP32**: Primary target for Edge AI inference
- **Arduino Nicla Sense ME**: Alternative with built-in IMU
- **BNO055**: 9-axis IMU sensor for motion capture

## Deployment

The trained RL models are quantized to 8-bit and deployed using TensorFlow Lite for Microcontrollers or ONNX Runtime for embedded systems.
