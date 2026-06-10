FROM flink:1.19-scala_2.12-java11

# Iceberg + Kafka + S3 connectors for the Flink runtime.
ENV ICEBERG_VERSION=1.6.0 FLINK_MAJOR=1.19
RUN cd /opt/flink/lib && \
    curl -fsSLO https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-flink-runtime-${FLINK_MAJOR}/${ICEBERG_VERSION}/iceberg-flink-runtime-${FLINK_MAJOR}-${ICEBERG_VERSION}.jar && \
    curl -fsSLO https://repo1.maven.org/maven2/org/apache/flink/flink-sql-connector-kafka/3.2.0-1.19/flink-sql-connector-kafka-3.2.0-1.19.jar && \
    curl -fsSLO https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-aws-bundle/${ICEBERG_VERSION}/iceberg-aws-bundle-${ICEBERG_VERSION}.jar

# PyFlink for submitting the Python job.
RUN pip install --no-cache-dir apache-flink==1.19.1
