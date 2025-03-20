# Use Python base image
FROM python:3.10
# Set working directory
WORKDIR /app
# Copy project files
COPY . .
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Expose the port Flask will run on
EXPOSE 5000
# Start the Flask app using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "bot:app"]