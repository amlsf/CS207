/**
 * @file poisson.cpp
 * Test script for treating the Graph as a MTL Matrix
 * and solving a Poisson equation.
 *
 * @brief Reads in two files specified on the command line.
 * First file: 3D Points (one per line) defined by three doubles.
 * Second file: Eges (one per line) defined by 2 indices into the point list
 *              of the first file.
 *
 * Launches an SDLViewer to visualize the solution.
 */

#include "CS207/SDLViewer.hpp"
#include "CS207/Util.hpp"

#include "Graph.hpp"
#include "Point.hpp"
#include "BoundingBox.hpp"
#include <fstream>
#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/itl/itl.hpp>
using namespace std;

bool is_Boundary(const Point&);

// HW3: YOUR CODE HERE
// Define a GraphSymmetricMatrix that maps
// your Graph concept to MTL's Matrix concept. This shouldn't need to copy or
// modify Graph at all!
typedef Graph<bool,char> GraphType;
typedef GraphType::Node Node;
typedef mtl::dense_vector<double> VectorType;

/** GraphSymmetricMatrix.
 * A abstract matrix that uses an underlying graph to represent the matrix.
 */
struct GraphSymmetricMatrix {

	/** GraphSymmetricMatrix Constructor.
	 * @param[in] g Graph that will be used to represent the Matrix.
	 */
	GraphSymmetricMatrix(GraphType& g) : g_(&g) {
		tagBoundaryNodes();
	}

	/** Helper function to perform multiplication. Allows for delayed
	 * 	evaluation of results.
	 * 	Assign::apply(a, b) resolves to an assignment operation such as
	 * 	a += b, a -= b, or a = b
	 * 	@pre @a size(v) == size(w)
	 */
	template <typename VectorIn, typename VectorOut, typename Assign>
	void mult(const VectorIn& v, VectorOut& w, Assign) const {
		assert(size(w) == size(v));

		for(size_t i = 0; i < g_->size(); ++i) {
			//Running sum
			double sum = 0;
			for(size_t j = 0; j < g_->size(); ++j) {
				sum += A(i, j) * v[j];
			}
			//Apply sum with vector out
			w[i] = Assign::apply( w[i], sum );
		}
	}

	/** Matrix-vector multiplication forwards to MTL's lazy mat_cvec_multiplier operation
	 */
	template <typename Vector>
	mtl::vec::mat_cvec_multiplier<GraphSymmetricMatrix, Vector> operator*(const Vector& v) const {
		return mtl::vec::mat_cvec_multiplier<GraphSymmetricMatrix, Vector>(*this, v);
	}

	/** Definition of A: Poisson with an implicit matrix.
	 * @param[in] i, j Indices that represent the row and column of a matrix.
	 * @return 1 if @a i == @a j && node is on a boundary
	 *         0 if @a i != @a j && (node @a i is on the boundary || @a i is on the boundary)
	 *         Laplacian(@a i, @a j), otherwise.
	 *
	 * Complexity: O(1)
	 */
	double A(size_t i, size_t j) const {
		if( i == j && g_->node(i).value() ) {
			return 1;
		}
		else if( i != j && (g_->node(i).value() || g_->node(j).value()) ) {
			return 0;
		}
		else {
			return Laplace(i, j);
		}
	}

	/** Definition of Laplacian operator.
	 * @param[in] i, j Indices that represent the row and column of a matrix.
	 * @return -node(i).degree() if @a i == @a j.
	 *         1 if has_edge(node(i), node(j)).
	 *         0 if otherwise.
	 *
	 * Complexity: O(1)
	 */
	double Laplace(size_t i, size_t j) const {
		if( i == j) {
			return -1.0 * g_->node(i).degree();
		}
		else if(g_->has_edge(g_->node(i), g_->node(j)) ) {
			return 1;
		}
		else {
			return 0;
		}
	}

	/** Returns Matrix Dimension
	 * @return graph.size()
	 */
	size_t get_MatrixDim() const {
		return g_->size();
	}

	/** Prints out contents in A
	 */
	void printA() {
		for(size_t i = 0; i < g_->size(); ++i) {
			for(size_t j = 0; j < g_->size(); ++j) {
				cout << A(i , j) << " ";
			}
			cout << endl;
		}
	}

	/** Prints out contents in A
	 */
	void printBoundaryNodesValues() {
		for(auto it = g_->node_begin(); it != g_->node_end(); ++it) {
			if((*it).value() ) {
				cout << 1;
			}
			else if(!(*it).value() ) {
				cout << 0;
			}
			else{
				cout << "NA";
			}
			cout << " ";
		}
		cout << endl;
	}

private:
	//Pointer to the graph
	GraphType* g_;

	//Private Helper Functions
	/** Uses is_Boundary function to tag boundary nodes.
	 */
	void tagBoundaryNodes() {
		for(auto it = g_->node_begin(); it != g_->node_end(); ++it) {
			if(is_Boundary((*it).position()) ) {
				(*it).value() = true;
			}
		}
	}


};

/** The number of elements in the matrix. */
inline std::size_t size(const GraphSymmetricMatrix& A) {
	return A.get_MatrixDim()*A.get_MatrixDim();
}

/** The number of rows in the matrix. */
inline std::size_t num_rows(const GraphSymmetricMatrix& A) {
	return A.get_MatrixDim();
}

/** The number of columns in the matrix. */
inline std::size_t num_cols(const GraphSymmetricMatrix& A) {
	return A.get_MatrixDim();
}

/** Traits that MTL uses to determine properties of our IdentityMatrix. */
namespace mtl {
namespace ashape {

/** Define IdentityMatrix to be a non-scalar type. */
template<>
struct ashape_aux<GraphSymmetricMatrix> {
	typedef nonscal type;
};
}

/** IdentityMatrix implements the Collection concept with value_type and size_type */
template<>
struct Collection<GraphSymmetricMatrix> {
	typedef double value_type;
	typedef unsigned size_type;
};
}

/** Finds if point is on the Boundary.
 * @param[in] Point x.
 * @return True if norm_inf(x) == 1.
 *         True if norm_inf(x - (+/-0.6, +/-0.6,0)) < 0.2.
 *         True if @a x is in the bouding box defined by (-0.6, -0.2, -1), (0.6, 0.2, 1).
 *		   False, otherwise.
 *
 * Complexity: O(1)
 */
bool is_Boundary(const Point& x) {
	const BoundingBox bb( Point(-0.6, -0.2, -1), Point(0.6, 0.2, 1) );

	if(norm_inf(x) == 1 ) {
		return true;
	}
	else if( norm_inf(x - Point(0.6, 0.6, 0)) < 0.2 ||
			norm_inf(x - Point(0.6, -0.6, 0)) < 0.2 ||
			norm_inf(x - Point(-0.6, 0.6, 0)) < 0.2 ||
			norm_inf(x - Point(-0.6, -0.6, 0)) < 0.2 ) {
		return true;
	}
	else if(bb.contains(x)) {
		return true;
	}
	else {
		return false;
	}
}

/** Define g(x) for Laplacian operator.
 * @param[in] Point x.
 * @return 0 if norm_inf(x) == 1.
 *         -0.2 if norm_inf(x - (+/-0.6, +/-0.6,0)) < 0.2.
 *         1 if @a x is in the bouding box defined by (-0.6, -0.2, -1), (0.6, 0.2, 1).
 *
 * Complexity: O(1)
 */
double g(const Point& x) {
	const BoundingBox bb( Point(-0.6, -0.2, -1), Point(0.6, 0.2, 1) );

	if(norm_inf(x) == 1 ) {
		return 0;
	}
	else if( norm_inf(x - Point(0.6, 0.6, 0)) < 0.2 ||
			norm_inf(x - Point(0.6, -0.6, 0)) < 0.2 ||
			norm_inf(x - Point(-0.6, 0.6, 0)) < 0.2 ||
			norm_inf(x - Point(-0.6, -0.6, 0)) < 0.2 ) {
		return -0.2;
	}
	else if(bb.contains(x)) {
		return 1;
	}
	else {
		return -1;
	}
}

/** Define f(x) for Laplacian operator.
 * @param[in] Point x.
 * @return 5*cos(norm_1(x)).

 * Complexity: O(1)
 */
double f(const Point& x) {
	return 5*cos(norm_1(x));
}

/** Remove all the nodes in graph @a g whose posiiton is contained within
 * BoundingBox @a bb
 * @post For all i, 0 <= i < @a g.num_nodes(),
 *        not bb.contains(g.node(i).position())
 */
void remove_box(GraphType& g, const BoundingBox& bb) {
	for(auto it = g.node_begin(); it != g.node_end();) {
		if( bb.contains( (*it).position() ) )
			g.remove_node(it);
		else
			++it;
	}
}

/** Creates a functor to color the matrix with patterns.
 */
struct makePatterns {

	CS207::Color operator()(const GraphType::Node& n) const {

		if(int(normSq(n.position()) * mult_) % 3 == 0) {
			return CS207::Color::make_hsv(0.2,1,1);
		}
		else if (int(normSq(n.position()) * mult_) % 3 == 1) {
			return CS207::Color::make_hsv(0.4,1,1);
		}
		else if (int(normSq(n.position()) * mult_) % 3 == 2) {
			return CS207::Color::make_hsv(0.6,1,1);
		}
		else {
			return CS207::Color::make_hsv(0.9,1,1);
		}
	}

	int mult_ = 100;
};

/** Functor that updates the position to the solution vector u.
 */
template<typename Vector>
struct UpdateSolution {
	Vector u_; //Solution vector

	/** UpdateSolution Constructor.
	 * @param[in] Vector solution vector.
	 */
	UpdateSolution(const Vector& u) : u_(u) {};

	/** UpdateSolution () Operator.
	 * @param[in] Valid node of the graph.
	 * @return a Point where the z corresponds to @a u_[n.index()] solution.
	 */
	Point operator()(const GraphType::Node& n) const {
		return Point(n.position().x, n.position().y, u_[n.index()]);
	}
};

//Define visual_iteration

namespace itl {
	/** visual_iteration Class.
	 */
	template <class Real, class ColorType, class PositionType, class OStream = std::ostream>
	class visual_iteration : public cyclic_iteration<Real> {
		typedef cyclic_iteration<Real> super;

		public:

		/** visual_iteration Constructor.
		 * @param[in] r0, max_iter_, tol_, atol_, cycle_ parameters for cyclic_iteration.
         * @param[in] Valid viewer, graph, solution_vector.
		 *
		 */
		template <class Vector>
		visual_iteration(const Vector& r0, int max_iter_, Real tol_, Real atol_, int cycle_,
				CS207::SDLViewer& viewer, GraphType& graph, VectorType& solution_vector,
				ColorType color_functor, PositionType position_functor, OStream& out = std::cout)
				: super(r0, max_iter_, tol_, atol_, cycle_),
				  viewer_(viewer), graph_(graph), solution_vector_(solution_vector),
				  color_functor_(color_functor), position_functor_(position_functor) {
			initializeViewer();
		}

		/** Update viewer until tolerance is reached.
		 * @returns True if super::finished().
		 */
		bool finished() {
			updateViewer();
			return super::finished();
		}

		/** Update viewer until tolerance is reached.
		 * @param[in] r Input into super::finished.
		 * @returns True if super::finished(r).
		 */
		template <typename T>
		bool finished(const T& r) {
			updateViewer();
			bool ret = super::finished(r);
			return ret;
		}

		private:
		CS207::SDLViewer& viewer_; //Viewer
		GraphType& graph_; //graph
		VectorType& solution_vector_; //solution vector
		ColorType color_functor_; //color functor
		PositionType position_functor_; //position functor
		std::map<typename GraphType::node_type, unsigned> node_map_; //node map for visualization

		/** initialize the Viewer.
		 */
		void initializeViewer(){
			node_map_ = viewer_.empty_node_map(graph_);
			viewer_.launch();
			updateViewer();
		}

		/** Update viewer with updated solutions and colors.
		 */
		void updateViewer() {
			viewer_.add_nodes(graph_.node_begin(), graph_.node_end(), color_functor_, UpdateSolution<VectorType>(solution_vector_), node_map_);
			viewer_.add_edges(graph_.edge_begin(), graph_.edge_end(), node_map_);
			viewer_.center_view();
		}

	};



}


int main(int argc, char** argv)
{
	// Check arguments
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " NODES_FILE EDGES_FILE\n";
		exit(1);
	}

	// Define an empty Graph
	GraphType graph;

	// Create a nodes_file from the first input argument
	std::ifstream nodes_file(argv[1]);
	// Interpret each line of the nodes_file as a 3D Point and add to the Graph
	std::vector<typename GraphType::node_type> node_vec;
	Point p;
	while (CS207::getline_parsed(nodes_file, p))
		node_vec.push_back(graph.add_node(2*p - Point(1,1,0)));

	// Create a tets_file from the second input argument
	std::ifstream tets_file(argv[2]);
	// Interpret each line of the tets_file as four ints which refer to nodes
	std::array<int,4> t;
	while (CS207::getline_parsed(tets_file, t)) {
		graph.add_edge(node_vec[t[0]], node_vec[t[1]]);
		graph.add_edge(node_vec[t[0]], node_vec[t[2]]);
		graph.add_edge(node_vec[t[1]], node_vec[t[3]]);
		graph.add_edge(node_vec[t[2]], node_vec[t[3]]);
	}



	// Get the edge length, should be the same for each edge
	double h = graph.edge(0).length();

	// Make holes in our Graph
	remove_box(graph, BoundingBox(Point(-0.8+h,-0.8+h,-1), Point(-0.4-h,-0.4-h,1)));
	remove_box(graph, BoundingBox(Point( 0.4+h,-0.8+h,-1), Point( 0.8-h,-0.4-h,1)));
	remove_box(graph, BoundingBox(Point(-0.8+h, 0.4+h,-1), Point(-0.4-h, 0.8-h,1)));
	remove_box(graph, BoundingBox(Point( 0.4+h, 0.4+h,-1), Point( 0.8-h, 0.8-h,1)));
	remove_box(graph, BoundingBox(Point(-0.6+h,-0.2+h,-1), Point( 0.6-h, 0.2-h,1)));

	// Print out the stats
	std::cout << graph.num_nodes() << " " << graph.num_edges() << std::endl;

	// HW3: YOUR CODE HERE
	// Define b using the graph, f, and g.
	// Construct the GraphSymmetricMatrix A using the graph
	// Solve Au = b using MTL.

	//Create A Matrix
	typedef GraphSymmetricMatrix  matrix_type;
	int N = graph.size();
	matrix_type A(graph);

	//Construct b and u
	VectorType b(N,0.0);
	VectorType u(N,0.0);

	//Fill in b with initial conditions
	for(int i = 0; i < N; ++i) {
		//If node i is on a boundary
		if(graph.node(i).value() ) {
			b[i] = g(graph.node(i).position());
		}
		//Not on boundary
		else {
			b[i] = h*h*f(graph.node(i).position());
			double result = 0;

			for(auto iIt = graph.node(i).edge_begin(); iIt != graph.node(i).edge_end(); ++iIt) {
				if( (*iIt).node2().value() ) {
					result += g((*iIt).node2().position() );
				}
			}
			b[i] -= result;
		}
	}

	cout << "b is " << b << endl;

	// Create an ILU(0) preconditioner
	itl::pc::identity<matrix_type> P(A);

	// Setup Viewer and UpdateSolution Functor
	UpdateSolution<VectorType> updateSol(u);
	CS207::SDLViewer viewer;

	// HW3 Problem 2
	// None real-time update
	//itl::cyclic_iteration<double> iter(b, 1000, 1.e-10, 0.0, 50);
	//cg(A, u, b, P, iter);

	// HW3 Problem 3
	// Real-time Update
	itl::visual_iteration<double, makePatterns, UpdateSolution<VectorType>>
		vis_iter(b, 1000, 1.e-10, 0.0, 50, viewer, graph, u, makePatterns(), updateSol);

	// Use Conjugate Gradient To Solve for Solution of A*u = b
	cg(A, u, b, P, vis_iter);

	return 0;
}
