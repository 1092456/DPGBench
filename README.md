# DPGBench
This is an evaluation framework code for graph synthesis methods, which includes both privacy evaluation and utility evaluation. The privacy evaluation consists of membership inference attacks (MIA) and attribute inference attacks (AIA), while the utility evaluation covers 15 commonly used metrics in graph synthesis methods.
The privacy evaluation module needs to be executed via the `run.sh` script using the following command:

```bash
./run.sh 
```

The utility evaluation module is configured directly within the code, and can be run simply with:

```bash
python run_utl.py
```
