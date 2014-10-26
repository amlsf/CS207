/**
 * @file subgraph.cpp
 * Test script for viewing a subgraph from our Graph
 *
 * @brief Reads in two files specified on the command line.
 * First file: 3D Points (one per line) defined by three doubles
 * Second file: Tetrahedra (one per line) defined by 4 indices into the point
 * list
 */

#include <fstream>
#include <iterator>

#include "CS207/SDLViewer.hpp"
#include "CS207/Util.hpp"

#include "Graph.hpp"
#include "math.h"

/** An iterator that skips over elements of another iterator based on whether
 * those elements satisfy a predicate.
 *
 * Given an iterator range [@a first, @a last) and a predicate @a pred,
 * this iterator models a filtered range such that all i with
 * @a first <= i < @a last and @a pred(*i) appear in order of the original range.
 */
template <typename Pred, typename It>
class filter_iterator
		: private equality_comparable<filter_iterator<Pred,It>> {
		public:
	// Get all of the iterator traits and make them our own
	typedef typename std::iterator_traits<It>::value_type        value_type;
	typedef typename std::iterator_traits<It>::pointer           pointer;
	typedef typename std::iterator_traits<It>::reference         reference;
	typedef typename std::iterator_traits<It>::difference_type   difference_type;
	typedef typename std::input_iterator_tag                     iterator_category;

	typedef filter_iterator<Pred,It> self_type;

	/** Filter Iterator Constructor.
	 * @param[in] p Predicate to all elements in @a first that return True.
	 * @param[in] first Iterator to first position to where @a p is called on.
	 * @param[in] last Iterator to the end position in @a first.
	 */
	filter_iterator(const Pred& p, const It& first, const It& last)
	: p_(p), it_(first), end_(last) {
		//Find the first it_ where p_ == True
		toValidPosition();
	}

	/* Dereference Filter Iterator and return the current position's edge.
	 * @pre Filter Iterator != @a first end.
	 * @return Valid value type to which the iterator contains.
	 *
	 * Complexity: O(*first).
	 */
	value_type operator*() const {
		return *it_;
	}

	/* Increment Filter Iterator and return the next position.
	 * @post Edge Iterator points to the next Edge position.
	 * @return Reference of this Filter Iterator that points to a new valid position determined by the predicate.
	 *
	 * Complexity: O(++first).
	 */
	self_type& operator++() {
		++it_;
		toValidPosition();
		return *this;
	}

	/* Compare two Filter Iterators to determine equivalence.
	 * @param[in] it Filter Iterator in graph.
	 * @return True if this Filter Iterator and @a it have the same iterator.
	 * False, otherwise.
	 *
	 * Complexity: O(first==).
	 */
	bool operator==(const self_type& it) const {
		return (it_ == it.it_);
	}

		private:
	/* Representative Invariants
	 * it_ == end_ || p(*it_)
	 */
	Pred p_;
	It it_;
	It end_;

	//Private helper function
	/** Fix the Representation Invariant and advance to a valid position.
	 * */
	void toValidPosition() {
		while (!p_(*it_) && it_!=end_)
			++it_;
	}
};

/** Helper function for constructing filter_iterators.
 *
 * Usage:
 * // Construct an iterator that filters odd values out and keeps even values.
 * std::vector<int> a = ...;
 * auto it = make_filtered(a.begin(), a.end(), [](int k) {return k % 2 == 0;});
 */
template <typename Pred, typename Iter>
filter_iterator<Pred,Iter> make_filtered(const Iter& it, const Iter& end,
		const Pred& p) {
	return filter_iterator<Pred,Iter>(p, it, end);
}

/** Test predicate for HW1 #4 */
struct SlicePredicate {
	template <typename NODE>
	bool operator()(const NODE& n) {
		return n.position().x < 0;
	}
};

/**Own predicate for HW1 #4 */
/** Predicate to remove area by removing nodes based on distance from Point(0,0,0).
 * This removes the bulk of the skull when applied to data/large.nodes.
 * @param[in] n A valid node.
 * @return True if distance measured from Point(0,0,0) to n.position() > 0.395.
 * Otherwise, false.
 *
 * Complexity: O(1)
 */
struct removeArea {
	double distance_ = 0.395;
	Point p_ = Point(0, 0, 0);
	template <typename NODE>
	bool operator()(const NODE& n) {
		return (sqrt(normSq(n.position() - p_)) > distance_ );
	}
};

int main(int argc, char** argv)
{
	// Check arguments
	if (argc < 3) {
		std::cerr << "Usage: " << argv[0] << " NODES_FILE TETS_FILE\n";
		exit(1);
	}

	// Construct a Graph
	typedef Graph<int, int> GraphType;
	GraphType graph;
	std::vector<GraphType::node_type> nodes;

	// Create a nodes_file from the first input argument
	std::ifstream nodes_file(argv[1]);
	// Interpret each line of the nodes_file as a 3D Point and add to the Graph
	Point p;
	while (CS207::getline_parsed(nodes_file, p))
		nodes.push_back(graph.add_node(p));

	// Create a tets_file from the second input argument
	std::ifstream tets_file(argv[2]);
	// Interpret each line of the tets_file as four ints which refer to nodes
	std::array<int,4> t;
	while (CS207::getline_parsed(tets_file, t))
		for (unsigned i = 1; i < t.size(); ++i)
			for (unsigned j = 0; j < i; ++j)
				graph.add_edge(nodes[t[i]], nodes[t[j]]);

	// Print out the stats
	std::cout << graph.num_nodes() << " " << graph.num_edges() << std::endl;

	// Launch the SDLViewer
	CS207::SDLViewer viewer;
	viewer.launch();

	// HW1 #4: YOUR CODE HERE
	// Use the filter_iterator to plot an induced subgraph.
	typedef removeArea predicateFunction;
	auto node_map = viewer.empty_node_map(graph);

	//Iterator initialization
	filter_iterator<predicateFunction, GraphType::NodeIterator> first =
			make_filtered(graph.node_begin(), graph.node_end(), predicateFunction() );
	filter_iterator<predicateFunction, GraphType::NodeIterator> last =
			make_filtered(graph.node_end(), graph.node_end(), predicateFunction() );

	//Add nodes and edges
	viewer.add_nodes(first, last, node_map);
	viewer.add_edges(graph.edge_begin(), graph.edge_end(), node_map);

	return 0;
}
