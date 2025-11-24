-   visualize how far the cat sees
-   fix scrollingj

Other tweakable settings:

-   number of angles
-   pattern dimension
-   number of inference iterations
-   normalization method
-   observation window size (number of observations in horizontally and vertically)
-   energy function variant
-   change hopfield.py to mhn.py
-   reame camera_simpulator to camera.py
-   rename localization_engine to localization.py
-   add how this maps to real world case into docs
-   how does angle averaging work?
-   z-index of heatmap positions?
-   which angle is shown when hovering?

Then also fix the settable parameters in the documentation.

Looking at the modern Hopfield network theory and your robot localization context, here are the key parameters you could expose in settings beyond beta:

## Additional Tunable Parameters

1. **Beta (β) - Temperature/Sharpness** _(already have)_

    - Controls the sharpness of the retrieval pattern

2. **Number of Stored Patterns (Memory Size)**

    - How many position-observation pairs to store
    - Affects memory capacity and retrieval accuracy

3. **Pattern Dimension**

    - Size of the observation vector (e.g., how many pixels/features)
    - Trade-off between detail and computational cost

4. **Update Iterations**

    - Number of energy minimization steps during retrieval
    - Modern Hopfield networks often converge in one step, but multiple iterations can help

5. **Normalization Method**

    - Whether to L2-normalize patterns before storage/retrieval
    - Affects the relative importance of pattern magnitude vs. direction

6. **Energy Function Variant**

    - Exponential vs. polynomial energy functions
    - Different convergence properties

7. **Similarity Threshold**

    - Minimum similarity score to accept a retrieval result
    - Reject ambiguous localizations below threshold

8. **Observation Window Size/Stride**
    - During memory building: how many observations to collect and spacing between them
    - Affects coverage density of the map

For your robot localization demo, I'd suggest focusing on:

-   **Beta** (already have)
-   **Memory size** (number of stored positions)
-   **Observation dimensions** (strip width/height)
-   **Similarity threshold** (confidence for accepting localization)
-   **Number of update iterations** (usually 1 is enough for modern Hopfield)

Would you like me to help you add any of these parameters to your settings dialog?
