FROM frolvlad/alpine-miniconda3 as build

WORKDIR pacpac
COPY environment.yml setup.py ./
WORKDIR pacpac
COPY pacpac .
WORKDIR ..

RUN conda env create -f environment.yml

RUN echo "source activate pacpac" > ~/.bashrc
ENV PATH /opt/conda/envs/pacpac/bin:$PATH

RUN pip install .

########## conda-pack bit to reduce image size
# Install conda-pack:
RUN conda install conda-pack

# Use conda-pack to create a standalone enviornment
# in /venv:
RUN conda-pack -n pacpac -o /tmp/env.tar && \
  mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
  rm /tmp/env.tar

# We've put venv in same path it'll be in final image,
# so now fix up paths:
RUN /venv/bin/conda-unpack

FROM debian:buster AS runtime

# Copy /venv from the previous stage:
COPY --from=build /venv /venv
##########

RUN echo "source /venv/bin/activate" > ~/.bashrc
ENV PATH /venv/bin:$PATH

ENTRYPOINT ["pacpac"]
