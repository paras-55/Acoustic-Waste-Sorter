
import time
import serial

PORT = "COM5"
BAUD = 115200

def main():
    print("Connecting to ESP32...")

    try:
        with serial.Serial(PORT, BAUD, timeout=2) as esp:
            time.sleep(2)
            esp.reset_input_buffer()

            print("Connected!")
            print("\nP = Plastic position")
            print("M = Metal position")
            print("C = Center position")
            print("Q = Quit")

            while True:
                command = input("\nEnter command: ").strip().upper()

                if command == "Q":
                    break

                if command not in ("P", "M", "C"):
                    print("Invalid command. Enter P, M, C, or Q.")
                    continue

                esp.write(command.encode("ascii"))
                esp.flush()

                response = esp.readline().decode(
                    "utf-8", errors="replace"
                ).strip()

                if response:
                    print("ESP32:", response)
                else:
                    print("No response received. Check the firmware and COM port.")

    except serial.SerialException as error:
        print("Connection error:", error)
        print("Close Arduino Serial Monitor and check COM5.")

    print("Controller closed.")

if __name__ == "__main__":
    main()
