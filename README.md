# PyLDT

This tiny library loads LDT files and calculates the photometric distribution.
You can extract this distribution as polar plots, pandas Dataframes, or just a matrix or flat array.

# Usage
```python
from pyldt import LDT

ldt = LDT.from_file("example.ldt")
other_ldt = LDT.from_raw_data(RAW_DATA)

ldt.compare_to(other_ldt)
#> 65 

ldt == other
#> False

ldt.dist
#> array([[2534.        , 2534.        , 2534.        , ..., 2534.        ,
#>         2534.        , 2534.        ],
#>         [2512.        , 2511.33333333, 2510.66666667, ..., 2507.6       ,
#>         2507.6       , 2507.6       ],
#>         [2490.        , 2488.66666667, 2487.33333333, ..., 2481.2       ,
#>         2481.2       , 2481.2       ],
#>         ...,
#>         [   0.        ,    0.        ,    0.        , ...,    0.        ,
#>         0.        ,    0.        ],
#>         [   0.        ,    0.        ,    0.        , ...,    0.        ,
#>         0.        ,    0.        ],
#>         [   0.        ,    0.        ,    0.        , ...,    0.        ,
#>         0.        ,    0.        ]])

ldt.max
#> 2536.0
ldt.min
#> 0.0

ldt.has_nan
#> False

ldt.as_array
#> array([2534., 2534., 2534., ...,    0.,    0.,    0.])

ldt.as_dataframe
#>  0            1            2       3    ...     357     358     359     360
#>  0    2534.0  2534.000000  2534.000000  2534.0  ...  2534.0  2534.0  2534.0  2534.0
#>  1    2512.0  2511.333333  2510.666667  2510.0  ...  2507.6  2507.6  2507.6  2507.6
#>  2    2490.0  2488.666667  2487.333333  2486.0  ...  2481.2  2481.2  2481.2  2481.2
#>  3    2468.0  2466.000000  2464.000000  2462.0  ...  2454.8  2454.8  2454.8  2454.8
#>  4    2446.0  2443.333333  2440.666667  2438.0  ...  2428.4  2428.4  2428.4  2428.4
#>  ..      ...          ...          ...     ...  ...     ...     ...     ...     ...
#>  176     0.0     0.000000     0.000000     0.0  ...     0.0     0.0     0.0     0.0
#>  177     0.0     0.000000     0.000000     0.0  ...     0.0     0.0     0.0     0.0
#>  178     0.0     0.000000     0.000000     0.0  ...     0.0     0.0     0.0     0.0
#>  179     0.0     0.000000     0.000000     0.0  ...     0.0     0.0     0.0     0.0
#>  180     0.0     0.000000     0.000000     0.0  ...     0.0     0.0     0.0     0.0
#>  [181 rows x 361 columns]

ldt.save_plot()
# Saves the current distribution as polar plot
```

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.
