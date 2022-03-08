FROM nvcr.io/nvidia/pytorch:22.01-py3

WORKDIR /app/

ADD requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

ADD . /app/

CMD ["--device=cuda"]
ENTRYPOINT [ "python", "examples/population.py" ]