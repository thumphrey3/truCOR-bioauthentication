#Using lightweight alpine image
FROM frolvlad/alpine-python-machinelearning

#Installing packages
RUN apk update
RUN apk add bash
RUN apk add bash-doc
RUN apk add bash-completion
RUN pip3 install flask
RUN pip3 install --no-cache-dir pipenv

#Define Working Directory
WORKDIR /usr/src/app
COPY Pipfile ./
COPY Pipfile.lock ./
COPY trubootstrap.sh ./
COPY trucor ./trucor
COPY blueberry.db ./

#Install API Dependencies
RUN pipenv --three

#Start app
EXPOSE 5000
ENTRYPOINT ["/usr/src/app/trubootstrap.sh"]
