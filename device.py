class Device:
    def __init__(self, name):
        self.name = name

        if self.name == "A40":
            self.memory_bandwidth = 696 * 2**30
            self.peak_performance = 149.7 * 10**12
        elif self.name == "Jetson Orin Nano 8GB":
            self.memory_bandwidth = 68.29 * 2**30
            self.peak_performance = 10 * 10**12
        else:
            raise ValueError("Name not supported in Device class")

        self.break_point = self.peak_performance / self.memory_bandwidth

    def __repr__(self):
        return (f"\n{'='*30}\n"
                f"Device Information:\n"
                f"{'-' * 30}\n"
                f" Name               : {self.name}\n"
                f" Memory Bandwidth   : {self.memory_bandwidth / 2**30:.2f} GB/s\n"
                f" Peak Performance   : {self.peak_performance / 10**12:.2f} TFLOP/s\n"
                f" Break Point        : {self.break_point:.2e} FLOPs/Byte\n"
                f"{'-' * 30}")
