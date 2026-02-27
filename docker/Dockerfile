FROM dolfinx/dolfinx:v0.9.0

RUN pip3 install --no-cache-dir --force-reinstall --no-deps \
    git+https://github.com/augustjohansson/2026-ffcx-priv.git@main

RUN pip3 install --no-cache-dir cppyy

RUN git clone --depth=1 https://github.com/algoim/algoim.git && \
    cp -r algoim /usr/include

RUN pip3 install --no-cache-dir \
    git+https://github.com/augustjohansson/2026-customquad-priv.git@main

RUN pip3 install --no-cache-dir pdbpp

RUN apt update && \
    apt install -y --no-install-recommends --no-install-suggests ccache && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
