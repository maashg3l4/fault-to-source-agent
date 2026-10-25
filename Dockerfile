# Use a lightweight version of Python 3.11
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the shopping list into the container
COPY requirements.txt .

# Install the Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# A default command to keep the container running (we will override this later)
CMD ["tail", "-f", "/dev/null"]