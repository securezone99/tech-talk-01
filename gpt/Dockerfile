FROM python:3.9-slim As base 

RUN apt-get update \
&&  apt-get install ffmpeg -y \
&& apt-get install -y --no-install-recommends git \
&& apt-get purge -y --auto-remove \
&& rm -rf /var/lib/apt/lists/*

# Add the needed files      
WORKDIR /gpt
ENV PROJECT_DIR=/gpt

# Create group and user called app 
RUN groupadd -r gpt -g 1000 \
 && useradd -u 1000 -r -g gpt -s /sbin/nologin -c "gpt user" gpt \
 && chown gpt:gpt /home --recursive 

# Install requirements for the User
COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

# Remove requirements.txt to clean environment 
RUN rm -rf requirements.txt 

COPY --chown=gpt src/ ${PROJECT_DIR}/src/

# Expose ports
EXPOSE 7000

USER gpt 

# Run the application
CMD [ "python" , "/gpt/src/controller/app.py" ]
