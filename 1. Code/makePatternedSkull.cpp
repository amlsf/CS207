/**
 * @file shortest_path.cpp
 * Test script for using our templated Graph to determine shortest paths.
 *
 * @brief Reads in two files specified on the command line.
 * First file: 3D Points (one per line) defined by three doubles
 * Second file: Tetrahedra (one per line) defined by 4 indices into the point
 * list
 */

#include <vector>
#include <fstream>

#include "CS207/SDLViewer.hpp"
#include "CS207/Util.hpp"
#include "CS207/Color.hpp"

#include "Graph.hpp"
#include <queue>
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
	 * @param[in] last Iterator to the end position in first.
	 */
	filter_iterator(const Pred& p, const It& first, const It& last)
	: p_(p), it_(first), end_(last) {
		//Find the first it_ where p_ == True
		toValidPosition();
	}

	/* Dereference Filter Iterator and return the current position's edge.
	 * @pre Filter Iterator != edge_end().
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
	// RI: it_ == end_ || p(*it_)
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
 * @return True if n.position > 0.395. Otherwise, false.
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

/** Comparator that compares the distance from a given point p.
 */
struct MyComparator {
	Point p_;
	/** Filter Iterator Constructor.
	 * @param[in] p Valid point.
	 */
	MyComparator(const Point& p) : p_(p) {
	};

	/** Filter Iterator Constructor.
	 * @param[in] node1 Valid first node.
	 * @param[in] node2 Valid second node.
	 * @return True if distance of node 1 from @a p's position is less than distance of node 2 from @a p's position.
	 *  Otherwise, false.
	 */
	template <typename NODE>
	bool operator()(const NODE& node1, const NODE& node2) const {
		return normSq(node1.position() - p_) < normSq(node2.position() - p_);
	}
};


/** Calculate shortest path lengths in @a g from the nearest node to @a point.
 * @param[in,out] g Input graph
 * @param[in] point Point to find the nearest node to.
 * @post Graph has modified node values indicating the minimum path length
 *           to the nearest node to @a point
 * @post Graph nodes that are unreachable to the nearest node to @a point have
 *           the value() -1.
 * @return The maximum path length found.
 *
 * Finds the nearest node to @a point and treats that as the root node for a
 * breadth first search.
 * This sets node's value() to the length of the shortest path to
 * the root node. The root's value() is 0. Nodes unreachable from
 * the root have value() -1.
 */
int shortest_path_lengths(Graph<int>& g, const Point& point) {

	//Find node to be Root node to start BFS
	Graph<int>::NodeIterator rootIt = std::min_element(g.node_begin(), g.node_end(), MyComparator(point) );

	//Initialize all nodes' values to be -1
	for(auto ni = g.node_begin(); ni != g.node_end(); ++ni) {
		(*ni).value() = -1;
	}

	//Define variables
	int longestPath = 0;
	(*rootIt).value() = 0;
	std::queue<Graph<int>::Node> nodes_q;
	nodes_q.push(*rootIt);

	//Begin BFS
	while(!nodes_q.empty() ) {
		//Set current Node
		Graph<int>::Node currNode = nodes_q.front();

		for(Graph<int>::IncidentIterator iit = currNode.edge_begin();
				iit != currNode.edge_end(); ++iit) {
			if( (*iit).node2().value() == -1 ){
				(*iit).node2().value() = currNode.value() + 1; //Note the distance
				nodes_q.push( (*iit).node2() ); //Push untraversed node into the queue
				if( longestPath < (*iit).node2().value() ) {
					longestPath = (*iit).node2().value();
				}
			}
		}
		//Pop off currNode
		nodes_q.pop();
	}
	return longestPath;
}

/** Creates a functor to color the skull to show where the shortest paths are.
 * @param[in] longestPath The longest path as determined from the shortest_path_lengths function.
 * @return A color functor that returns a heat map that shows where the longest path regions are.
 * Red denotes the longest paths and blue denotes the shortest paths.
 */
typedef CS207::Color Color;
struct MyColorFunctor {
	float longestPath_ = 1.0;
	MyColorFunctor(const float& longestPath) : longestPath_(longestPath) {};

	CS207::Color operator()(const Graph<int>::Node& n) const {
		return CS207::Color::make_heat(n.value()/longestPath_);
	}
};

/** Creates a functor to color the skull with patterns.
 * @param[in] longestPath The longest path as determined from the shortest_path_lengths function.
 * @return A color functor that colors the skull yellow, green, blue based on modular functions.
 *  Also shows the longest path regions by brightening them with dark regions having the shortest paths.
 */
struct makePatterns {
	float longestPath_ = 1.0;
	makePatterns(const float& longestPath) : longestPath_(longestPath) {};

	Color operator()(const Graph<int>::Node& n) const {
		if(n.value() % 3 == 0) {
			return Color::make_hsv(0.2,1,n.value()/longestPath_);
		}
		else if (n.value() % 3 == 1) {
			return Color::make_hsv(0.4,1,n.value()/longestPath_);
		}
		else if (n.value() % 3 == 2) {
			return Color::make_hsv(0.6,1,n.value()/longestPath_);
		}
		else {
			return Color::make_hsv(0.9,1,n.value()/longestPath_);
		}
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
	typedef Graph<int> GraphType;
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
	// Use shortest_path_lengths to set the node values to the path lengths
	// Construct a Color functor and view with the SDLViewer
	int longestPath = shortest_path_lengths(graph, Point(-1,0,1));
	std::cout << std::endl;
	std::cout << "LongestPath: " << longestPath << std::endl;
	std::cout << std::endl;

	//Colors
	makePatterns colorFunctor = makePatterns(longestPath);

	//Subgraph
	typedef removeArea predicateFunction;
	//Iterator initialization
	filter_iterator<predicateFunction, Graph<int>::NodeIterator> first =
			make_filtered(graph.node_begin(), graph.node_end(), predicateFunction() );
	filter_iterator<predicateFunction, Graph<int>::NodeIterator> last =
			make_filtered(graph.node_end(), graph.node_end(), predicateFunction() );

	auto node_map = viewer.empty_node_map(graph);
	viewer.add_nodes(first, last, colorFunctor, node_map);
	viewer.add_edges(graph.edge_begin(), graph.edge_end(), node_map);

	return 0;
}




