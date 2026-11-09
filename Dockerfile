FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir '.[inference,hub,structure]'
RUN useradd --create-home evotcr
USER evotcr
WORKDIR /home/evotcr
ENTRYPOINT ["evotcr"]
CMD ["--help"]
