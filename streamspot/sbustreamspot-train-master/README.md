# StreamSpot Train

<img src="http://www3.cs.stonybrook.edu/~emanzoor/streamspot/img/streamspot-logo.jpg" height="150" align="right"/>

[http://www3.cs.stonybrook.edu/~emanzoor/streamspot/](http://www3.cs.stonybrook.edu/~emanzoor/streamspot/)

## Requirements

   * [Anaconda](https://www.continuum.io/downloads) for the entire `scikit-learn` stack.
   * GCC 5.2+ to compile the C++11 code.

## Training Procedure

The following steps assume this repository has been cloned and all dependencies installed.

**Convert the training data from CDM13/Avro to StreamSpot**

For detailed instructions, see the [sbustreamspot-cdm README](https://github.com/sbustreamspot/sbustreamspot-cdm).

For the purpose of instruction, `infoleak_small_units.CDM13.avro` is assumed to be the training data.

   * Get the StreamSpot CDM translation code: `git clone https://github.com/sbustreamspot/sbustreamspot-cdm.git`
   * Install its dependencies: `pip install -r requirements.txt`
   * Convert CDM13/Avro training data to StreamSpot edges: `python translate_cdm_to_streamspot.py --url avro/infoleak_small_units.CDM13.avro --format avro --source file --concise > streamspot/infoleak_small_units.CDM13.ss`

**Convert the StreamSpot training graphs to shingle vectors**

The graph-to-shingle-vector transformation code is in C++ to ensure high performance.
It is a modified version of the [streamspot-core](https://github.com/sbustreamspot/sbustreamspot-core) code.

Build and run the code as follows;
```
cd graphs-to-shingle-vectors
make optimized
./streamspot --edges=../streamspot/infoleak_small_units.CDM13.ss --chunk-length 24 > ../shingles/infoleak_small_units.CDM13.sv
cd ..
```

**Cluster the training graph shingle vectors**

Ensure the dependencies have been installed: `pip install -r requirements.txt`
```
python create_seed_clusters.py  --input shingles/infoleak_small_units.CDM13.sv > clusters/infoleak_small_units.CDM13.cl
```

The `*.cl` file can then be provided to [streamspot-core](https://github.com/sbustreamspot/sbustreamspot-core).

## Contact

   * emanzoor@cs.stonybrook.edu
   * leman@cs.stonybrook.edu
