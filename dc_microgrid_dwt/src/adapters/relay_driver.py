import logging

class RelayDriver:
    def open_relay(self):
        logging.getLogger("RelayDriver").critical("OPENING RELAY - CIRCUIT ISOLATED")
        # GPIO.output(PIN, HIGH)

    def close_relay(self):
        logging.getLogger("RelayDriver").info("Closing Relay - Circuit Restored")
