FROM nvcr.io/nvidia/pytorch:23.07-py3


WORKDIR /app

COPY . .

# no-cache: 
RUN pip install -r requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT [ "bash", "run_linux.sh" ]
