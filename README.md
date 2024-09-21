# JSON Neighbor

Simple CLI tool for finding the closest and furthest matches of JSON files containing numeric data.

## Usage

```sh
poetry run json-neighbor --help
```

```sh
$ poetry run json-neighbor example_files/file1.json example_files -n 2

Closest files:
- example_files\file3.json (distance: 1.415)
- example_files\file4.json (distance: 2.740)

Furthest files:
- example_files\file2.json (distance: 4.302)
- example_files\file4.json (distance: 2.740)
```
