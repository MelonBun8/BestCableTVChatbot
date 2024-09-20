#dockerfile, Image, Container

FROM python:3.12.4

ADD main.py .
ADD query_assigner.py .
ADD sql_helper.py .


