def elapsed_time_to_string(elapsed_time: float) -> str:
    """
    Generate a string formatted in hours:minutes:seconds given the elapsed time in seconds.
    :param elapsed_time: The elapsed time in seconds.
    :return:
    """
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    # Format the elapsed time
    time_str = f"{int(hours):02}h:{int(minutes):02}m:{seconds:.2f}s"
    return time_str
