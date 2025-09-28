# Use Python 3.9 base
FROM python:3.9

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 1) Work as root (default)
WORKDIR /app

# 2) Copy only requirements first for better layer caching
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# 3) Create the non-root user BEFORE switching to it
RUN useradd -m -u 1000 user
ENV HOME=/home/user

# 4) Prepare user home and app dir, set ownership
RUN mkdir -p $HOME/app && chown -R user:user $HOME


# 5) Now switch to the non-root user
USER user
WORKDIR $HOME/app

# 6) Copy the rest of the source as that user
COPY --chown=user:user . $HOME/app


# 7) Run your app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
