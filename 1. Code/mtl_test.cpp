/**
 * @file mtl_test.cpp
 * Test script for interfacing with MTL4 and it's linear solvers.
 */

// HW3: Need to install/include Boost and MTL in Makefile
#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/itl/itl.hpp>
#include<iostream>
using namespace std;
#include <cassert>

// HW3: YOUR CODE HERE
// Define a IdentityMatrix that interfaces with MTL
struct IdentityMatrix {

	/** IdentityMatrix Constructor.
	 * @param[in] matrix_dim
	 */
	IdentityMatrix(size_t matrix_dim) : matrix_dim_(matrix_dim) {}

	/** Helper function to perform multiplication. Allows for delayed
	 * 	evaluation of results.
	 * 	Assign::apply(a, b) resolves to an assignment operation such as
	 * 	a += b, a -= b, or a = b
	 * 	@pre @a size(v) == size(w)
	 */
	template <typename VectorIn, typename VectorOut, typename Assign>
	void mult(const VectorIn& v, VectorOut& w, Assign) const {
		assert(size(w) == size(v));
		for(size_t i = 0; i < size(w); ++i) {
			w[i] = Assign::apply( w[i], v[i] );
		}
	}

	/** Matrix-vector multiplication forwards to MTL's lazy mat_cvec_multiplier operation
	 */
	template <typename Vector>
	mtl::vec::mat_cvec_multiplier<IdentityMatrix, Vector> operator*(const Vector& v) const {
		return mtl::vec::mat_cvec_multiplier<IdentityMatrix, Vector>(*this, v);
	}

	/** Compute the product of a vector with this identity matrix
	 * HW3 Part 1: Commented out
	 template <typename Vector>
		Vector operator*(const Vector& x) const {
			return x;
		}
	 */

	/** Returns Matrix Dimension
	 */
	size_t get_MatrixDim() const {
		return matrix_dim_;
	}
private:
	//Matrix dimension
	size_t matrix_dim_;

};

/** The number of elements in the matrix. */
inline std::size_t size(const IdentityMatrix& A) {
	return A.get_MatrixDim()*A.get_MatrixDim();
}

/** The number of rows in the matrix. */
inline std::size_t num_rows(const IdentityMatrix& A) {
	return A.get_MatrixDim();
}

/** The number of columns in the matrix. */
inline std::size_t num_cols(const IdentityMatrix& A) {
	return A.get_MatrixDim();
}

/** Traits that MTL uses to determine properties of our IdentityMatrix. */
namespace mtl {
namespace ashape {

/** Define IdentityMatrix to be a non-scalar type. */
template<>
struct ashape_aux<IdentityMatrix> {
	typedef nonscal type;
};
}

/** IdentityMatrix implements the Collection concept with value_type and size_type */
template<>
struct Collection<IdentityMatrix> {
	typedef double value_type;
	typedef unsigned size_type;
};
}


int main()
{
	// HW3: YOUR CODE HERE
	// Construct an IdentityMatrix and "solve" Ix = b
	// using MTL's conjugate gradient solver

	//Create Identity Matrix
	const int size = 10;
	typedef IdentityMatrix  matrix_type;
	matrix_type A(size);

	// Create an ILU(0) preconditioner
	itl::pc::identity<matrix_type> P(A);

	// Set b such that x == 1 is solution; start with x == 0
	mtl::dense_vector<double> X(size, 1.0), b(size);
	b = A * X;
	X = 0;

	cout << endl;
	cout << "Before Solving" << endl;
	cout << "x is " << X << endl;
	cout << "b is " << b << endl;

	//Use Conjugate Gradient To Solve for Solution
	itl::cyclic_iteration<double> iter(b, 100, 1.e-11, 0.0, 5);
	cg(A, X, b, P, iter);

	cout << endl;
	cout << "After Solving" << endl;
	cout << "x is " << X << endl;
	cout << "b is " << b << endl;
	cout << endl;

	//Check to see if Conjugate Gradient Worked
	//X should be equal to b
	if(X == b)
		cout << "X and b are equal" << endl;
	else
		cout << "X and b are not equal" << endl;

	return 0;
}
