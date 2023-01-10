/*
 * Copyright 2020-2023 OpenDR European Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Eigen-decomposition for symmetric 3x3 real matrices.
   Public domain, copied from the public domain Java library JAMA. */

#ifndef _eig_h

/* Symmetric matrix A => eigenvectors in columns of V, corresponding
   eigenvalues in d. */
void eigen_decomposition(double A[3][3], double V[3][3], double d[3]);

#endif
