# what is cosine similarity?
- Cosine similarity measures the similarity between two vectors by calculating the cosine of the angle between them, ranging from -1 to 1.
- Cosine similarity quantifies how closely two vectors align, regardless of their magnitude, by focusing on the angle between them. Mathematically, for two vectors A and B, the formula is:
`cosine similarity = (A · B) / (||A|| × ||B||)`

Where:

- A · B is the dot product of vectors A and B, calculated by multiplying corresponding components and summing the results.
- ||A|| and ||B|| are the magnitudes (lengths) of vectors A and B, computed as the square root of the sum of the squares of their components.

The resulting value ranges from -1 to 1:
- 1 indicates the vectors point in the same direction (identical orientation).
- 0 indicates the vectors are orthogonal (no similarity).
- -1 indicates the vectors point in opposite directions (completely dissimilar) ( +2).

Example :
- Consider vectors x = [3, 2, 0, 5] and y = [1, 0, 0, 0].

- Compute the dot product: x · y = (3×1) + (2×0) + (0×0) + (5×0) = 3

Compute magnitudes:
- ||x|| = √(3² + 2² + 0² + 5²) = √(9 + 4 + 0 + 25) = √38 ≈ 6.16
- ||y|| = √(1² + 0² + 0² + 0²) = √1 = 1

- Cosine similarity: cosine_sim = 3 / (6.16 × 1) ≈ 0.49
- The dissimilarity can also be expressed as: 1 - cosine similarity = 0.51

Applications

Cosine similarity is widely used in:

- Text analysis: comparing documents or sentences by representing them as term frequency vectors.
- Machine learning: clustering, recommendation systems, and measuring similarity between feature vectors.
- Information retrieval: ranking search results based on similarity to a query 
This formula is particularly useful because it ignores vector magnitude, making it ideal for comparing data of different scales or lengths.
