import datetime

def MyFun(date_str):
    # Convert input string to datetime object
    date_obj = datetime.datetime.strptime(date_str, "%d-%m")

    # Get day of year as an integer
    day_of_year = date_obj.timetuple().tm_yday

    # Return day of year
    return day_of_year