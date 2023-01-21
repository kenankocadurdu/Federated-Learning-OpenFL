#!/bin/bash
set -e

fx envoy start -n director_client --disable-tls --envoy-config-path envoy_config.yaml -dh localhost -dp 50051
