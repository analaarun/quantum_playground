# %%
import time
import streamlit as st
st.set_page_config(layout="wide")

from datetime import datetime
import pytz

# %%

timezones = [
    ('India Time (IST)', 'Asia/Kolkata'),
    ('Canada Time (EST)', 'Canada/Eastern'),
    ('San Francisco Time (PST)', 'America/Los_Angeles'),
    ('Dallas Time (CST)', 'America/Chicago'),
    ('Eastern Time (EST)', 'US/Eastern'),
    ('GMT Time', 'GMT')
]

current_times = []
ist_timezone = pytz.timezone('Asia/Kolkata')

for label, tz in timezones:
    timezone = pytz.timezone(tz)
    # Create a naive UTC time
    naive_now = datetime.utcnow()
    
    # Localize the time to UTC and then convert to target timezone
    utc_time = pytz.UTC.localize(naive_now)
    current_time = utc_time.astimezone(timezone)
    
    # Calculate time difference with IST
    ist_time = utc_time.astimezone(ist_timezone)
    time_difference = (current_time.utcoffset() - ist_time.utcoffset()).total_seconds() / 3600
    
    current_times.append((label, current_time, time_difference))

# Sort times in ascending order
current_times.sort(key=lambda x: x[1])

# Display times
for label, current_time, time_difference in current_times:
    formatted_time = current_time.strftime('%a, %d %b %Y %I:%M:%S %p %Z')
    if time_difference > 0:
        difference_text = f"{abs(time_difference):.1f} hours ahead of IST"
    elif time_difference < 0:
        difference_text = f"{abs(time_difference):.1f} hours behind IST"
    else:
        difference_text = "same as IST"
    st.markdown(f"<h2 style='color: lightblue; background-color: black;'>{label}: {formatted_time} ({difference_text})</h2>", unsafe_allow_html=True)


placeholder = st.empty()

# Replace the placeholder with some text:
placeholder.text("Hello")

# Replace the text with a chart:
# placeholder.line_chart({"data": [1, 5, 2, 6]})

# Replace the chart with several elements:
with placeholder.container():
     st.write("This is one element")
     st.write("This is another")

# Clear all those elements:
placeholder.empty()