Project to build a model capable of predicting whether or not an employee will remain working at a company or leave.

Please refer to the file `Employee-Attrition.ipynb` for the code and visualization that goes along with these projects. The csv files can be found in the `Data` directory.

To run a metaflow pipeline of the chosen model, return

```{bash}
python flow.py show #to see the DAG of the pipeline
python flow.py run #to run the code
```

`python flow.py show` should product the following graph:


```{bash}
Step start
    ?
    => clean_data

Step clean_data
    ?
    => calculate_vif_

Step calculate_vif_
    ?
    => calculate_corr_

Step calculate_corr_
    ?
    => predict

Step predict
    ?
    => metrics

Step metrics
    ?
    => end

Step end
    ?
```
