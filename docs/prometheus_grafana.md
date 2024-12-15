# Prometheus and Grafana

To install prometheus, download from [here](https://prometheus.io/download/) and follow the instructions below.

```bash
tar xvf prometheus-*.tar.gz
cd prometheus-*
```
You can find sample `prometheus.yml` in the `prometheus-*` directory. There is a default one in [here](./testing/prometheus.yml).
start prometheus
```bash
./prometheus --config.file=prometheus.yml
```
start the app
```bash
python host.py
```
check `http://localhost:8000/metrics` to see if the metrics are being scraped.


visit `http://localhost:3000` to start create dashboards.

* TODO: integrate with ROS2 topics
