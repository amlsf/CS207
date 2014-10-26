#ifndef CS207_GRAPH_HPP
#define CS207_GRAPH_HPP

/** @file Graph.hpp
 * @brief An undirected graph type
 */
#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>

#include "CS207/Util.hpp"
#include "Point.hpp"



/** @class Graph
 * @brief A template for 3D undirected graphs.
 *
 * Users can add and retrieve nodes and edges. Edges are unique (there is at
 * most one edge between any pair of distinct nodes).
 */
template <typename V, typename E>
class Graph {
private:
	struct internal_node_element;

public:

	/////////////////////////////
	// PUBLIC TYPE DEFINITIONS //
	/////////////////////////////

	/** Type of this graph. */
	typedef Graph graph_type;

	/** Predeclaration of Node type. */
	class Node;
	/** Synonym for Node (following STL conventions). */
	typedef Node node_type;

	/** Predeclaration of Edge type. */
	class Edge;
	/** Synonym for Edge (following STL conventions). */
	typedef Edge edge_type;

	/** Type of indexes and sizes.
      Return type of Graph::Node::index(), Graph::num_nodes(),
      Graph::num_edges(), and argument type of Graph::node(size_type) */
	typedef unsigned size_type;

	/** Type of node iterators, which iterate over all graph nodes. */
	class NodeIterator;
	/** Synonym for NodeIterator */
	typedef NodeIterator node_iterator;

	/** Type of edge iterators, which iterate over all graph edges. */
	class EdgeIterator;
	/** Synonym for EdgeIterator */
	typedef EdgeIterator edge_iterator;

	/** Type of incident iterators, which iterate incident edges to a node. */
	class IncidentIterator;
	/** Synonym for IncidentIterator */
	typedef IncidentIterator incident_iterator;

	/** Synomym for V*/
	typedef V node_value_type;

	/** Synomym for E*/
	typedef E edge_value_type;

	////////////////////////////////
	// CONSTRUCTOR AND DESTRUCTOR //
	////////////////////////////////

	/** Construct an empty graph. */
	Graph() : vec_Nodes_(), adj_list_Edges_(), num_edges_(0), i2u_(){
		//Initialize to empty containers
	}
	/** Default destructor */
	~Graph() = default;

	/////////////
	// General //
	/////////////

	/** Return the size of the graph.
	 * @return The number of nodes s.t. size() == num_nodes() && size() >= 0.
	 * Complexity: O(1).
	 */
	size_type size() const {
		return i2u_.size();
	}

	/** Remove all nodes and edges from this graph.
	 * @post num_nodes() == 0 && num_edges() == 0
	 *
	 * Invalidates all outstanding Node and Edge objects.
	 */
	void clear() {
		vec_Nodes_.clear();
		adj_list_Edges_.clear();
		i2u_.clear();
		num_edges_ = 0;
	}

	/////////////////
	// GRAPH NODES //
	/////////////////

	/** @class Graph::Node
	 * @brief Class representing the graph's nodes.
	 *
	 * Node objects are used to access information about the Graph's nodes.
	 */
	class Node : private totally_ordered<Node> {
	public:
		/** Construct an invalid node.
		 *
		 * Valid nodes are obtained from the Graph class, but it
		 * is occasionally useful to declare an @i invalid node, and assign a
		 * valid node to it later. For example:
		 *
		 * @code
		 * Graph::node_type x;
		 * if (...should pick the first node...)
		 *   x = graph.node(0);
		 * else
		 *   x = some other node using a complicated calculation
		 * do_something(x);
		 * @endcode
		 */
		Node() {
		}

		/** Return this node's position.
		 * @return The node's Point object
		 *
		 * Complexity O(1)
		 */
		const Point& position() const {
			return graph_->vec_Nodes_[uid_].p_;
		}

		/** Return this node's position.
		 * @return The node's Point object
		 *
		 * Complexity O(1)
		 */
		Point& position() {
			return graph_->vec_Nodes_[uid_].p_;
		}

		/** Return this node's index, a number in the range [0, graph_size).
		 * @return The node's index i, s.t. 0 <= i < num_nodes()
		 * Complexity O(1)
		 * */
		size_type index() const {
			return graph_->vec_Nodes_[uid_].idx_;
		}

		/** Test whether this node and @a x are equal.
		 * @param[in] @a x is a node
		 * @return True if this node has the same graph pointer and index;
		 * otherwise False.
		 *
		 * Complexity O(1)
		 */
		bool operator==(const Node& x) const {
			return ((graph_ == x.graph_) && (uid_ == x.uid_));
		}

		/** Test whether this node is less than @a x in the global order.
		 * @param[in] @a x is a node
		 * @return True if this node's graph address is less than @a x's graph address;
		 * or node's graph address is the same as @a x's, then this node's index is less
		 * than @a x's index.
		 *  False, otherwise.
		 *
		 * Complexity O(1)
		 */
		//Use the node's index to determine operator<
		bool operator<(const Node& x) const {
			if(graph_ != x.graph_) {
				return (graph_ < x.graph_);
			}
			else{
				return (uid_ < x.uid_);
			}
		}

		//Prints out contents in an node
		void print_node() const {
			std::cout << "graph address: " << graph_ << std::endl;
			std::cout << "n uid: " << uid_ << std::endl;
			std::cout << "n position: " << position() << std::endl;
			std::cout << "n value: " << value() << std::endl;
			std::cout << "n index: " << index() << std::endl;
		}

		/** Get this node's value (modifiable).
		 * @return This node's node_value_type value as a reference.
		 *
		 * Complexity O(1)
		 */
		node_value_type& value() {
			return graph_->vec_Nodes_[uid_].value_;
		}
		/** Get this node's value (non-modifiable).
		 * @return This node's node_value_type value as a const reference (non-modifiable).
		 *
		 * Complexity O(1)
		 */
		const node_value_type& value() const {
			return graph_->vec_Nodes_[uid_].value_;
		}

		/** Get the number of incident edges connected to this node.
		 * @return The number of incident edges connected to this node,
		 * s.t. 0 <= degree() < num_nodes().
		 *
		 * Complexity O(1)
		 */
		size_type degree() const {
			return graph_->adj_list_Edges_[uid_].size();
		}

		/** Set an Incident Iterator to the start of this node's incident edges.
		 * @return Iterator that points to the start of this node's incident edges.
		 *
		 * Complexity O(1)
		 */
		incident_iterator edge_begin() const {
			return IncidentIterator(graph_, uid_, 0 );
		}

		/** Set an Incident Iterator to the end of this node's incident edges.
		 * @return Iterator that points to the first of this node's invalid incident edges.
		 *
		 * Complexity O(1)
		 */
		incident_iterator edge_end() const {
			return IncidentIterator(graph_, uid_, degree() );
		}

		//Prints out all the Incident Edges to this node using an Incident Iterator
		void print_incidentEdges_w_iter() const {
			std::cout << "Fixed Node: " << uid_ << std::endl;
			int count = 0;
			for (auto iit = edge_begin(); iit != edge_end(); ++iit) {
				std::cout << "Node: " << uid_ << std::endl;
				std::cout << "Incident Edge: "<< count++ << std::endl;
				std::cout << (*iit).node1().position() << std::endl;
				std::cout << (*iit).node2().position() << std::endl;
				std::cout << "Value: " << (*iit).value() << std::endl;
			}
		}

	private:
		// Allow Graph to access Node's private member data and functions.
		friend class Graph;

		//private member variables
		/* Representative Invariants
		 * g_ != nullptr
		 * 0 <= uid_ < num_unique_nodes()
		 */
		Graph* graph_; //Pointer to the parent graph
		size_type uid_; //uid for the node in the parent graph

		/** Private constructor for Graph to construct nodes
		 * @param[in] graph This new node's parent graph.
		 * @param[in] uid The uID that @a graph uses to uniquely identify this node. @a uid >= 0.
		 */
		Node(const Graph* graph, size_type uid)
		: graph_(const_cast<Graph*>(graph)), uid_(uid) {}

	};

	/** Return the number of nodes in the graph.
	 * @return The number of nodes s.t. num_nodes() >= 0.
	 *
	 * Complexity: O(1).
	 */
	size_type num_nodes() const {
		return i2u_.size();
	}

	/** Return the number of unique nodes in the graph.
	 * @return The number of nodes s.t. num_unique_nodes() >= 0.
	 *
	 * Complexity: O(1).
	 */
	size_type num_unique_nodes() const {
		return vec_Nodes_.size();
	}

	/** Add a node to the graph, returning the added node.
	 * @param[in] position The new node's position
	 * @post new size() == old size() + 1
	 * @post result_node.index() == old size()
	 * @return Node with index() == old_size() and uid_ == old_num_unique_nodes.size()
	 *
	 * Complexity: O(1) amortized operations.
	 */
	Node add_node(const Point& position,
			const node_value_type& value = node_value_type()) {
		internal_node_element temp_ele;
		temp_ele.p_ = position;
		temp_ele.value_ = value;
		temp_ele.idx_ = i2u_.size();
		//Add to vector containers
		vec_Nodes_.push_back(temp_ele);
		i2u_.push_back( vec_Nodes_.size() - 1);
		//resize Adj list
		adj_list_Edges_.resize( vec_Nodes_.size() );
		adj_list_Edges_values_.resize( vec_Nodes_.size() );
		return Node(this, vec_Nodes_.size() - 1);
	}

	/** Determine if this Node belongs to this Graph
	 * @return True if @a n is currently a Node of this Graph
	 *
	 * Complexity: O(1).
	 */
	bool has_node(const Node& n) const {
		if(this == n.graph_ && n.uid_ < num_unique_nodes()) {
			return true;
		}
		else {
			return false;
		}
	}

	/** Return the node with index @a i.
	 * @pre 0 <= @a i < num_nodes()
	 * @post result_node.index() == i
	 *
	 * Complexity: O(1).
	 */
	Node node(size_type i) const {
		assert(i < this->num_nodes());
		return Node(this, i2u_[i]);
	}

	/** Remove the node and its associated edges from the graph
	 * param[in] n Valid node of this Graph.
	 * @pre has_node(@a n) == true.
	 * @pre graph.node(i).index() == i.
	 * @post graph.node(i).index() == i.
	 * @post graph.node(@a n.index()) == @a n.
	 * @post has_node(@a n) == false.
	 * @post new num_nodes() = old num_nodes() - 1.
	 * @post Edges with e.node1() == @a n or e.node2() == @a n are invalid and removed from the Graph.
	 * @post new num_edges() = old num_edges() - degree().
	 * @post Invalidates outstanding NodeIterators
	 * @post Invalidates outstanding EdgeIterators
	 * @post Invalidates outstanding IncidentIterators that are iterating through incident edges on @a n.
	 * @return size_type The index of the node when it is removed from the i2u_ vector.
	 *  size_type == i2u_.size() if the node was at the end.
	 *
	 * Complexity: Average case: O(degree()^2) and Worst case: O(num_nodes()^2).
	 */
	size_type remove_node(const Node& n) {
		size_type result = 0;
		//Erase
		for(unsigned i = 0; i < i2u_.size(); ++i) {
			if(i2u_[i] == n.uid_) {
				result = i;
				break;
			}
		}

		//Erase node
		i2u_.erase(i2u_.begin() + result);

		//Erase corresponding edges
		//while the array is not empty, keep erasing
		while( !adj_list_Edges_[n.uid_].empty() ) {
			size_type n2_uid = adj_list_Edges_[n.uid_][0];
			remove_edge( n, Node(n.graph_, n2_uid) );
		}

		//Reindex
		for(unsigned i = result; i < i2u_.size(); ++i) {
			internal_node_element& n = vec_Nodes_.at(i2u_[i]);
			n.idx_ = i;
		}
		return result;
	}

	/** Remove the node and its associated edges from the graph via NodeIterator
	 * param[in] n Valid NodeIterator of this Graph.
	 * @pre has_node(@a n) == true.
	 * @pre graph.node(i).index() == i.
	 * @post graph.node(i).index() == i.
	 * @post graph.node(@a n.index()) == @a n.
	 * @post has_node(@a n) == false.
	 * @post new num_nodes() = old num_nodes() - 1.
	 * @post Edges with e.node1() == @a n or e.node2() == @a n are invalid and removed from the Graph.
	 * @post new num_edges() = old num_edges() - degree().
	 * @post Invalidates outstanding NodeIterators
	 * @post Invalidates outstanding EdgeIterators
	 * @post Invalidates outstanding IncidentIterators that are iterating through incident edges on @a n.
	 * @return NodeIterator that points to the position of where the node was when it is removed from the i2u_ vector.
	 *  size_type == node_end() if the node was at the end.
	 *
	 * Complexity: Average case: O(degree()^2) and Worst case: O(num_nodes()^2).
	 */
	node_iterator remove_node(node_iterator n_it) {
		Node n = *n_it;
		int index = remove_node(n);
		return NodeIterator(this, index);
	}

	//Print out all of the nodes in ever stored in vector
	void print_vec_Node() const {
		for(unsigned i = 0; i < vec_Nodes_.size(); i++) {
			const internal_node_element& n = vec_Nodes_.at(i);
			std::cout << "node uid: " << i << std::endl;
			std::cout << "node point: " << n.p_ << std::endl;
			std::cout << "node value: " << n.value_ << std::endl;
			std::cout << "node index: " << n.idx_ << std::endl;
			std::cout << std::endl;
		}
	}

	//Print out all of the valid nodes in vector
	void print_indexed_Node() const {
		for(unsigned i = 0; i < i2u_.size(); i++) {
			const internal_node_element& n = vec_Nodes_.at(i2u_[i]);
			std::cout << "node index: " << i << std::endl;
			std::cout << "node point: " << n.p_ << std::endl;
			std::cout << "node value: " << n.value_ << std::endl;
			std::cout << "node index: " << n.idx_ << std::endl;
			std::cout << std::endl;
		}
	}

	/////////////////
	// GRAPH EDGES //
	/////////////////

	/** @class Graph::Edge
	 * @brief Class representing the graph's edges.
	 *
	 * Edges are order-insensitive pairs of nodes. Two Edges with the same nodes
	 * are considered equal if they connect the same nodes, in either order.
	 */
	class Edge : private totally_ordered<Edge> {
	public:
		/** Construct an invalid Edge. */
		Edge() {
		}

		/** Return the first node of this Edge
		 * return This edge's first node s.t. the Node is in the same graph
		 * and index is the unique ID for the node.
		 *
		 * Complexity: O(1).
		 */
		Node node1() const {
			return Node(graph_, uid1_);
		}

		/** Return the second node of this Edge
		 * return This edge's second node s.t. the Node is in the same graph
		 * and index is the unique ID for the node.
		 *
		 * Complexity: O(1).
		 */
		Node node2() const {
			return Node(graph_, uid2_);
		}

		/** Test whether this edge and @a x are equal.
		 * @param[in] x Edge in a graph
		 * @return True if this Edge's graph pointer is the same as @a x's graph pointer &&
		 * both nodes' uids match.
		 *
		 * Equal edges are from the same graph and have the same nodes.
		 *
		 * Complexity: O(1).
		 */
		bool operator==(const Edge& x) const {
			return ((graph_ == x.graph_) && (node1() == x.node1()) &&
					(node2() == x.node2()));
		}

		/** Test whether this edge is less than @a x in the global order.
		 * @param[in] x Edge in a graph
		 * @return True if this Edge's first node index is less than @a x's first node index;
		 * Or if both Edges have the same first node indices, Edge's second node index is less than
		 * @a x's second node; Or if both Edges have the same two node indices, this Edge's graph is
		 * less than @a x's graph address.
		 *  False, otherwise.
		 *
		 * Complexity: O(1).
		 */
		//Use graph pointer and index to determine operator <
		bool operator<(const Edge& x) const {
			if(uid1_ < x.uid1_){
				return true;
			}
			else if(uid1_ == x.uid1_ && uid2_ < x.uid2_){
				return true;
			}
			else if(uid1_ == x.uid1_ && uid2_ == x.uid2_ && graph_ < x.graph_){
				return true;
			}
			else {
				return false;
			}
		}

		/** Return the length of the Edge
		 * @pre Both nodes have valid positions.
		 * @return Double length between the two nodes by Euclidean distance formula.
		 *
		 * Complexity: O(1).
		 */
		double length() const {
			return norm( node1().position() - node2().position() );
		}

		/** Store or retrieve the Edge's value
		 * @pre Valid Edge.
		 * @return reference to this Edge's value.
		 *
		 * Complexity: O(d).
		 */
		edge_value_type& value() {
			Node n;
			//Swap edges
			Edge e;
			if(node1().uid_ > node2().uid_) {
				e = Edge(graph_, node2().uid_, node1().uid_);
			}
			else{
				e = Edge(graph_, node1().uid_, node2().uid_);
			}
			Node n1;
			Node n2;
			n1 = e.node1();
			n2 = e.node2();

			size_type list_index2;
			//Look for where the second uid is in the Adj list
			for(IncidentIterator it = n1.edge_begin(); it != n1.edge_end(); ++it) {
				if((*it).uid2_ == e.uid2_) {
					list_index2 = it.list_index2_;
				}
			}
			return graph_->adj_list_Edges_values_[e.uid1_][list_index2];
		}

		/** Retrieve the Edge's value (Cannot be modified)
		 * @pre Valid Edge.
		 * @return reference to this Edge's value.
		 *
		 * Complexity: O(d).
		 */
		const edge_value_type& value() const {
			Node n;
			//Swap edges
			Edge e;
			if(node1().uid_ > node2().uid_) {
				e = Edge(graph_, node2().uid_, node1().uid_);
			}
			else{
				e = Edge(graph_, node1().uid_, node2().uid_);
			}
			Node n1;
			Node n2;
			n1 = e.node1();
			n2 = e.node2();

			size_type list_index2;
			//Look for where the second uid is in the Adj list
			for(IncidentIterator it = n1.edge_begin(); it != n1.edge_end(); ++it) {
				if((*it).uid2_ == e.uid2_) {
					list_index2 = it.list_index2_;
				}
			}
			return graph_->adj_list_Edges_values_[e.uid1_][list_index2];
		}

		//Prints out contents in an edge
		void print_edge() const {
			std::cout << "graph address: " << graph_ << std::endl;
			std::cout << "n1 index: " << uid1_ << std::endl;
			std::cout << "n1 position: " << node1().position() << std::endl;
			std::cout << "n2 index: " << uid2_ << std::endl;
			std::cout << "n2 position: " << node2().position() << std::endl;
		}


	private:
		// Allow Graph to access Edge's private member data and functions.
		friend class Graph;
		/* Representative Invariants
		 * g_ != nullptr
		 * 0 <= uid1_ < num_unique_nodes()
		 * 0 <= uid2_ < num_unique_nodes()
		 */
		Graph* graph_; //Pointer to the parent graph for the edge
		size_type uid1_; //uid1 for the first node in the parent graph
		size_type uid2_; //uid2 for the second node in the parent graph

		/** Private constructor for Graph to construct edges.
		 * @param[in] graph This new edge's parent graph.
		 * @param[in] index1 The uID that @a graph uses to access this Edge's first node in the adjacency list. @a uid1 >= 0.
		 * @param[in] index1 The uID that @a graph uses to access this Edge's second node in the adjacency list. @a uid2 >= 0.
		 */
		Edge(const Graph* graph, size_type uid1, size_type uid2)
		: graph_(const_cast<Graph*>(graph)), uid1_(uid1), uid2_ (uid2) {}
	};

	/** Return the total number of edges in the graph.
	 * @return The number of edges s.t. 0 <= num_edges() <= Combination(num_nodes(), 2).
	 *
	 * Complexity: O(1).
	 */
	//Requirement: Complexity: No more than O(num_nodes() + num_edges()), hopefully less
	size_type num_edges() const {
		return num_edges_;
	}

	/** Add an edge to the graph, or return the current edge if it already exists.
	 * @pre @a a and @a b are distinct valid nodes of this graph
	 * @return an Edge object e with e.node1() == @a a and e.node2() == @a b
	 * @post has_edge(@a a, @a b) == true
	 * @post If old has_edge(@a a, @a b), new num_edges() == old num_edges().
	 *       Else,                        new num_edges() == old num_edges() + 1.
	 *
	 * Can invalidate edge indexes -- in other words, old edge(@a i) might not
	 * equal new edge(@a i). Must not invalidate outstanding Edge objects.
	 *
	 * Complexity: O(d) where d is @a a.degree().
	 */
	//Requirement: Complexity: No more than O(num_nodes() + num_edges()), hopefully less
	Edge add_edge(const Node& a, const Node& b) {
		//Make sure the user doesn't add the same nodes to an edge
		assert(a.uid_ != b.uid_ );

		//Check to see if graph has edge
		if(!has_edge(a, b)){
			//Graph does not have this edge so add this Edge to graph
			set_adj_list_edges(a.uid_, b.uid_);
			++num_edges_;
			return Edge(this, a.uid_, b.uid_);
		}

		//Graph already contains edges so return edge
		return Edge(this, a.uid_, b.uid_);
	}

	/** Test whether two nodes are connected by an edge.
	 * @pre @a a and @a b are valid nodes of this graph
	 * @return true if, for some @a i, edge(@a i) connects @a a and @a b.
	 *
	 * Complexity: O(d) where d is @a a.degree().
	 */
	//Requirement: Complexity: No more than O(num_nodes() + num_edges()), hopefully less
	bool has_edge(const Node& a, const Node& b) const {
		size_type node1_index = a.uid_;
		size_type node2_index = b.uid_;
		//Search for Edge in adjacency list
		for(unsigned i = 0; i < adj_list_Edges_[node1_index].size(); i++){
			if(adj_list_Edges_[node1_index][i] == node2_index)
				return true;
		}
		//Did not find Edge in adjacency list
		return false;
	}

	/** Return the edge with index @a i.
	 * @pre 0 <= @a i < num_edges()
	 * @return Edge with same graph pointer and associated nodes' indices
	 * where i is the ith valid edge in the adjacency list.
	 *
	 * Complexity: O(num_edges()).
	 */
	// Requirement: Complexity: No more than O(num_nodes() + num_edges()), hopefully less
	Edge edge(size_type i) const {
		assert(i < this->num_edges());
		EdgeIterator it = edge_begin();
		for( ; i != 0; --i) {
			++it;
		}
		return Edge(this, (*it).uid1_, (*it).uid2_);
	}

	/** Return Edge from this Graph.
	 * @param[in] Nodes @a n1 and @a n2 that represent an Edge.
	 * @pre Valid nodes @a n1 and @a n2 of this graph.
	 * @post new num_edge() = old num_edge() - 1 if has_Edge(n1, n2) == True.
	 * @post new num_edge() = old num_edge() if has_Edge(n1, n2) == False.
	 * @post EdgeIterators are invalidated.
	 * @post IncidentIterators are invalidated.
	 * @return size_type List index where the second node is removed from the Adj list.
	 * Returns @a n1's old num_degree() if the has_Edge(n1, n2) == False.
	 *
	 * Complexity: O(degree()).
	 */
	size_type remove_edge(const Node& n1, const Node& n2) {

		//First removal
		size_type result;
		result = n1.degree();
		bool edgeExists = false;
		for(auto iit = n1.edge_begin(); iit != n1.edge_end(); ++iit) {
			if( (*iit).uid2_ == n2.uid_) {
				adj_list_Edges_[n1.uid_].erase(adj_list_Edges_[n1.uid_].begin() + iit.list_index2_);
				adj_list_Edges_values_[n1.uid_].erase(adj_list_Edges_values_[n1.uid_].begin() + iit.list_index2_);
				edgeExists = true;
				result = iit.list_index2_;
				break;
			}
		}
		//Second removal
		for(auto iit2 = n2.edge_begin(); iit2 != n2.edge_end(); ++iit2) {
			if( (*iit2).uid2_ == n1.uid_) {
				adj_list_Edges_[n2.uid_].erase(adj_list_Edges_[n2.uid_].begin() + iit2.list_index2_);
				adj_list_Edges_values_[n2.uid_].erase(adj_list_Edges_values_[n2.uid_].begin() + iit2.list_index2_);
				edgeExists = true;
				break;
			}
		}

		//Decrement the number of edges if the edgeExists in the adjacency list
		if(edgeExists)
			--num_edges_;
		return result;
	}

	/** Return Edge from this Graph.
	 * @param[in] Edge e.
	 * @pre Valid edge @a e of this graph.
	 * @post new num_edge() = old num_edge() - 1 if has_Edge(n1, n2) == True.
	 * @post new num_edge() = old num_edge() if has_Edge(n1, n2) == False.
	 * @post EdgeIterators are invalidated.
	 * @post IncidentIterators are invalidated.
	 * @return size_type List index where the second node is removed from the Adj list.
	 * Returns @a n1's old num_degree() if the has_Edge(n1, n2) == False.
	 *
	 * Complexity: O(degree()).
	 */
	size_type remove_edge(const Edge& e) {
		return remove_edge( e.node1(), e.node2() );
	}

	/** Return Edge from this Graph.
	 * @param[in] EdgeIterator.
	 * @pre Valid EdgeIterator of this graph.
	 * @post new num_edge() = old num_edge() - 1 if has_Edge(n1, n2) == True.
	 * @post new num_edge() = old num_edge() if has_Edge(n1, n2) == False.
	 * @post EdgeIterators are invalidated.
	 * @post IncidentIterators are invalidated.
	 * @return EdgeIterator that points to the position of the next valid edge in the Adj list.
	 * Returns the graph.edge_end() if the removed edge is the last edge or there was no edge
	 * in the graph.
	 *
	 * Complexity: O(degree()).
	 */
	edge_iterator remove_edge(edge_iterator e_it) {
		Edge e = (*e_it);
		size_type list_index = remove_edge( e.node1(), e.node2() );
		return EdgeIterator(this, e.node1().uid_, list_index);
	}

	//Prints out this Graph's Edges' and Nodes' contents in the adjacency list.
	void print_adj_list() const {
		std::cout << "Start to print valid adj list" << std::endl;

		for(unsigned i = 0; i < adj_list_Edges_.size(); i++) {
			for(unsigned k = 0; k < i2u_.size(); k++) {
				if(i == i2u_[k]) {
					std::cout << "Node " << i << ": ";
					for( unsigned j = 0; j < adj_list_Edges_[i].size(); j++) {
						std::cout << adj_list_Edges_[i][j] << " ";
					}
					std::cout << std::endl;
				}
			}
		}
	}

	//Prints out this Graph's Edges' and Nodes' contents in the adjacency list.
	void print_all_adj_list() const {
		std::cout << "Start to print all adj list" << std::endl;
		for(unsigned i = 0; i < adj_list_Edges_.size(); i++) {
			std::cout << "Node " << i << ": ";
			for( unsigned j = 0; j < adj_list_Edges_[i].size(); j++) {
				std::cout << adj_list_Edges_[i][j] << " ";
			}
			std::cout << std::endl;
		}
	}

	//Prints out this Graph's Edges' and Nodes' contents in the adjacency list values.
	void print_adj_list_values() const {
		std::cout << "Start to print valid adj list values" << std::endl;

		for(unsigned i = 0; i < adj_list_Edges_.size(); i++) {
			for(unsigned k = 0; k < i2u_.size(); k++) {
				if(i == i2u_[k]) {
					std::cout << "Node " << i << ": ";
					for( unsigned j = 0; j < adj_list_Edges_values_[i].size(); j++) {
						std::cout << adj_list_Edges_values_[i][j] << " ";
					}
					std::cout << std::endl;
				}
			}
		}
	}

	///////////////
	// Iterators //
	///////////////

	/** @class Graph::NodeIterator
	 * @brief Iterator class for nodes. A forward iterator. */
	class NodeIterator : private totally_ordered<NodeIterator> {
	public:
		// These type definitions help us use STL's iterator_traits.
		/** Element type. */
		typedef Node value_type;
		/** Type of pointers to elements. */
		typedef Node* pointer;
		/** Type of references to elements. */
		typedef Node& reference;
		/** Iterator category. */
		typedef std::input_iterator_tag iterator_category;
		/** Difference between iterators */
		typedef std::ptrdiff_t difference_type;

		/** Construct an invalid NodeIterator. */
		NodeIterator() {
		}

		/* Dereference NodeIterator and return the current position's node.
		 * @pre NodeIterator != node_end().
		 * @return Valid node with this Graph's pointer and unique ID as index.
		 *
		 * Complexity: O(1).
		 */
		Node operator*() const {
			assert(graph_ != nullptr && index_ >= 0 );
			return Node(graph_, graph_->i2u_[index_]);
		}

		/* Increment NodeIterator and return the next position.
		 * @post NodeIterator points to the next node position (could be valid or invalid position, user needs to check).
		 * @return Reference of this NodeIterator that points to the new position = old position + 1.
		 *
		 * Complexity: O(1).
		 */
		node_iterator& operator++() {
			assert(graph_ != nullptr && index_ >= 0 );
			index_++;
			return *this;
		}

		/* Compare two NodeIterator to determine equivalence.
		 * @param[in] ni NodeIterator in graph.
		 * @return True if this NodeIterator and @a ni have the same graph pointer and same position in the graph denoted by the node's unique ID.
		 * False, otherwise.
		 * Complexity: O(1).
		 */
		bool operator==(const NodeIterator& ni) const {
			return (graph_ == ni.graph_ && index_ == ni.index_);
		}

	private:
		friend class Graph;
		//Private member variables
		/* Representative Invariants
		 * g_ != nullptr
		 * 0 <= index_ < num_nodes()
		 */
		Graph* graph_; //Pointer to the parent graph
		size_type index_; //index for the node in the parent graph

		/** Private constructor for Graph to construct Node Iterators.
		 * @param[in] graph This new edge's parent graph.
		 * @param[in] index The index that denotes the position of the NodeIterator in the @a graph, s.t. index >= 0.
		 */
		NodeIterator(const Graph* graph, size_type index)
		: graph_(const_cast<Graph*>(graph)), index_(index) {}
	};

	/** Set a Node Iterator to the start of this Graph's nodes.
	 * @return Node Iterator that points to the start of this graph's nodes.
	 * The position == node_end() if num_nodes() == 0.
	 *
	 * Complexity O(1)
	 */
	node_iterator node_begin() const {
		return NodeIterator(this, 0);
	}

	/** Set a Node Iterator to the end of this Graph's nodes.
	 * @return Node Iterator that points to the this graph's first invalid nodes.
	 * The position == node_begin() if num_nodes() == 0.
	 *
	 * Complexity O(1)
	 */
	node_iterator node_end() const {
		return NodeIterator(this, num_nodes());
	}

	//Print out this Graph's node content with a NodeIterator
	void print_Node_w_iter() {
		int count = 0;
		for (auto ni = node_begin(); ni != node_end(); ++ni) {
			std::cout << "Node: "<< count++ << std::endl;
			std::cout << "Degrees: " << (*ni).degree() << std::endl;
			std::cout << (*ni).position() << std::endl;
			std::cout << (*ni).value() << std::endl;
		}
	}

	/** @class Graph::EdgeIterator
	 * @brief Iterator class for edges. A forward iterator. */
	class EdgeIterator : private totally_ordered<EdgeIterator> {
	public:
		// These type definitions help us use STL's iterator_traits.
		/** Element type. */
		typedef Edge value_type;
		/** Type of pointers to elements. */
		typedef Edge* pointer;
		/** Type of references to elements. */
		typedef Edge& reference;
		/** Iterator category. */
		typedef std::input_iterator_tag iterator_category;
		/** Difference between iterators */
		typedef std::ptrdiff_t difference_type;

		/** Construct an invalid EdgeIterator. */
		EdgeIterator() {
		}

		/* Dereference Edge Iterator and return the current position's edge.
		 * @pre EdgeIterator != edge_end().
		 * @return Valid Edge with this Graph's pointer and unique IDs associated with the nodes.
		 *
		 * Complexity: O(1).
		 */
		Edge operator*() const {
			assert(graph_ != nullptr && n_index1_ < graph_->adj_list_Edges_.size() &&
					graph_->adj_list_Edges_[n_index1_].size() != 0);
			return Edge(graph_, n_index1_, graph_->adj_list_Edges_[n_index1_][list_index_]);
		}

		/* Increment Edge Iterator and return the next position.
		 * @post Edge Iterator points to the next Edge position.
		 * @return Reference of this Edge Iterator that points to the new position = old position + i where i = number of invalid positions.
		 *
		 * Complexity: O(1).
		 */
		EdgeIterator& operator++() {
			assert(graph_ != nullptr && n_index1_ < graph_->adj_list_Edges_.size() &&
					graph_->adj_list_Edges_[n_index1_].size() != 0);
			increment_n_index1();
			toValidPosition();
			return *this;
		}

		/* Compare two Edge Iterators to determine equivalence.
		 * @param[in] ei Edge Iterator in graph.
		 * @return True if this Edge Iterator and @a ei have the same graph pointer && point to the same edge denoted by the two nodes' unique IDs.
		 * False, otherwise.
		 *
		 * Complexity: O(1).
		 */
		bool operator==(const EdgeIterator& ei) const {
			return (graph_ == ei.graph_ && n_index1_ == ei.n_index1_ &&
					list_index_ == ei.list_index_ );
		}

	private:
		friend class Graph;
		//Private member variables
		/* Representative Invariants
		 * g_ != nullptr
		 * n_index1_ < graph_->adj_list_Edges_.size()
		 * n_index1_ < graph_->adj_list_Edges[n_index1_][list_index_]
		 * End: n_index1 == num_unique_nodes()-1 && list_index_ == adj_list_Edges[n_index1_].size()-1
		 */
		Graph* graph_; //Pointer to the parent graph
		size_type n_index1_; //index for the node 1
		size_type list_index_; //index for adjacency list

		/** Private constructor for Graph to construct Edge Iterator.
		 * @param[in] graph This new Edge Iterator's parent graph.
		 * @param[in] n_index1 The uID that denotes the position of the Edge Iterator's edge's first node in the @a graph, s.t. 0<= n_index < num_unique_nodes().
		 * @param[in] list_index The index that denotes the position of the list index in the adjacency list to access the Edge Iterator's edge's second node in the @a graph.
		 * s.t. 0<= list_index < graph->adj_list_Edges[n_indexd1].size().
		 */
		EdgeIterator(const Graph* graph, size_type n_index, size_type list_index)
		: graph_(const_cast<Graph*>(graph)), n_index1_(n_index), list_index_(list_index) {
			//Find the first valid position
			toValidPosition();
		}

		//Private Helper functions

		/** Private helper function to increment n_index1 and list_index_ to traverse through the adjacency list.
		 */
		void increment_n_index1() {
			//Current node contains no edges, skip to next n_index1_
			if(graph_->adj_list_Edges_[n_index1_].size() == 0 ){
				++n_index1_;
				list_index_ = 0;
			}
			//Reached the end of the edges in the node, skip to next n_index1_
			else if(list_index_ == graph_->adj_list_Edges_[n_index1_].size() -1 ) {
				++n_index1_;
				list_index_ = 0;
			}
			//More elements in adj_list_Edge_[n_index1_] to traverse, increment list_index_
			else {
				++list_index_;
			}
		}

		/** Private helper function to increment advance Edge Iterator to a valid Position.
		 *  A valid position is defined s.t. the iterator is not at the end of the adjacency list &&
		 *  graph_->adj_list_Edges_[first index].size() > 0 &&
		 *  for all i,j, edge(i, j), j>i.
		 *  Otherwise, increment until valid position is reached.
		 */
		void toValidPosition() {
			//Check to see if iterator is at the end
			while(!isEnd() ) {
				if(graph_->adj_list_Edges_[n_index1_].size() > 0) {
					//Reached valid position if for all i,j, edge(i, j), j>i
					if(graph_->adj_list_Edges_[n_index1_][list_index_] > n_index1_) {
						break;
					}
				}
				//Increment until a valid position is reached
				increment_n_index1();
			}
		}

		/** Private helper function to determine if the Edge Iterator is at the end of the adjacency list.
		 * @return True if num_unique_nodes() == 0
		 * Or n_index1_ == num_unique_nodes()-1 && list_index_ == adj_list_Edges[num_unique_nodes()-1].size()-1
		 */
		bool isEnd() {
			//Boundary case when there are no nodes in the graph
			if(graph_->num_unique_nodes() == 0) {
				return true;
			}
			//Boundary case where the end of the adjacency corresponds to a node with no edges
			else if(graph_->adj_list_Edges_[graph_->num_unique_nodes()-1].size() == 0) {
				return (n_index1_ == graph_->num_unique_nodes()-1 &&
						list_index_ == 0 );
			}
			//General case
			else {
				return (n_index1_ == graph_->num_unique_nodes()-1 &&
						list_index_ == graph_->adj_list_Edges_[graph_->num_unique_nodes()-1].size()-1 );
			}
		}

		//Print out Edge Iterator's private member variables
		void printPrivate() const {
			std::cout << "N index: " << n_index1_ << std::endl;
			std::cout << "List index: " << list_index_ << std::endl;
		}

	};

	/** Set an Edge Iterator to the start of this graph's edges.
	 * @return Edge Iterator that points to the start of this graphs edges, Edge Iterator == edge_end() if num_edges() == 0 || num_unique_nodes() == 0.
	 *
	 * Complexity: O(1).
	 */
	edge_iterator edge_begin() const {
		return EdgeIterator(this, 0, 0 );
	}

	/** Set an Edge Iterator to the end of this graph's edges.
	 * @return Edge Iterator that points to the first of this graphs's invalid edges, Edge Iterator == edge_begin() if num_edges() == 0 || num_unique_nodes() == 0.
	 *
	 * Complexity: O(1).
	 */
	edge_iterator edge_end() const {
		//Boundary case where there are no nodes in the graph
		if (num_unique_nodes() == 0 ) {
			return EdgeIterator(this, 0, 0 );
		}
		else{
			//Boundary case where the end of the adjacency corresponds to a node with no edges
			if(this->adj_list_Edges_[num_unique_nodes()-1].size() == 0) {
				return EdgeIterator(this, num_unique_nodes()-1 , 0 );
			}
			//General case
			else {
				return EdgeIterator(this, num_unique_nodes()-1 , this->adj_list_Edges_[num_unique_nodes()-1].size() - 1 );
			}
		}
	}

	//Print out this Graph's contents of all its Edges.
	void print_Edge_w_iter() {
		int count = 0;
		for (auto ei = edge_begin(); ei != edge_end(); ++ei) {
			std::cout << "Edge: "<< count++ << std::endl;
			std::cout << "NODE1: " << (*ei).node1().position() << std::endl;
			std::cout << "NODE2: " << (*ei).node2().position() << std::endl;
		}
		std::cout << "Finished printing Edges" << std::endl;
	}

	/** @class Graph::IncidentIterator
	 * @brief Iterator class for edges incident to a node. A forward iterator. */
	class IncidentIterator : private totally_ordered<IncidentIterator> {
	public:
		// These type definitions help us use STL's iterator_traits.
		/** Element type. */
		typedef Edge value_type;
		/** Type of pointers to elements. */
		typedef Edge* pointer;
		/** Type of references to elements. */
		typedef Edge& reference;
		/** Iterator category. */
		typedef std::input_iterator_tag iterator_category;
		/** Difference between iterators */
		typedef std::ptrdiff_t difference_type;

		/** Construct an invalid IncidentIterator. */
		IncidentIterator() {
		}

		/* Dereference Incident Iterator and return the current position's edge.
		 * @pre Incident Iterator != node.edge_end().
		 * @return Valid Edge with this Graph's pointer and unique IDs associated with the nodes.
		 *
		 * Complexity: O(1).
		 */
		Edge operator*() const {
			assert(graph_ != nullptr && uid1_ < graph_->num_unique_nodes() && list_index2_ < graph_->num_nodes() );
			return Edge(graph_, uid1_, graph_->adj_list_Edges_[uid1_][list_index2_]);
		}

		/* Increment Incident Iterator and return the next position.
		 * @post Incident Iterator points to the next incident edge's position (could be valid or invalid position, user needs to check).
		 * @return Reference of this Incident Iterator that points to the new position = old position + 1.
		 *
		 * Complexity: O(1).
		 */
		IncidentIterator& operator++() {
			assert(graph_ != nullptr && list_index2_ < graph_->num_nodes() );
			++list_index2_;
			return *this;
		}

		/* Compare two Incident Iterators to determine equivalence.
		 * @param[in] iit Incident Iterator in graph.
		 * @return True if this Incident Iterator and @a iit have the same graph pointer && point to the same edge denoted by the nodes' unique IDs.
		 * False, otherwise.
		 * Complexity: O(1).
		 */
		bool operator==(const IncidentIterator& iit) const {
			return (graph_ == iit.graph_ && uid1_ == iit.uid1_ && list_index2_ == iit.list_index2_);
		}
	private:
		friend class Graph;
		//Private member variables
		/* Representative Invariants
		 * g_ != nullptr
		 * 0 <= index1_ < num_unique_nodes()
		 * 0 <= list_index2_ < num_nodes()
		 */
		Graph* graph_; //Pointer to the parent graph for the edge
		size_type uid1_; //uid1 for the edge's first node in the parent graph
		size_type list_index2_; //index2 list index to access edge's second node in the adj list

		/** Private constructor for Graph to construct Incident Iterator.
		 * @param[in] graph This new Incident Iterator's parent graph.
		 * @param[in] uid1 The uID that denotes the position of the Incident Iterator's edge's first node in the @a graph, s.t. uid1 >= 0.
		 * @param[in] list_index2_ The index that denotes the position of the Incident Iterator's edge's second node in the @a graph, s.t. index2 >= 0.
		 */
		IncidentIterator(const Graph* graph, size_type uid1, size_type list_index2)
		: graph_(const_cast<Graph*>(graph)), uid1_(uid1), list_index2_ (list_index2) {}

	};

	private:
	//Internal structure used to store node elements
	struct internal_node_element{
		Point p_;
		node_value_type value_;
		size_type idx_;
	};

	//Internal structure used to store edge elements
	struct internal_edge_element{
		edge_value_type value;
	};

	//Private member variables
	//Container to store node elements
	std::vector<internal_node_element> vec_Nodes_;
	//Container to save adjacency list for Edges
	std::vector<std::vector<size_type>> adj_list_Edges_;
	//Keep track of number of edges in the graph
	int num_edges_;
	//Container to save Edge values
	std::vector<std::vector<edge_value_type>> adj_list_Edges_values_;
	//index to unique id conversion
	std::vector<size_type> i2u_;

	//Private Helper Functions

	/** Add new Edge to adjacency list.
	 * @param[in] uid1 uID that represents this Edge's first node.
	 * @param[in] uid2 uID that represents this Edge's second node.
	 *
	 * @pre @a index1 and @a index are >= 0.
	 * @post adj_list_Edges_[uid1].size() + 1.
	 *       adj_list_Edges_[uid2].size() + 1.
	 *       adj_list_Edges_values_[uid1].size() + 1.
	 *       adj_list_Edges_values_[uid2].size() + 1.
	 *
	 * Complexity: O(1) amortized operations.
	 */
	void set_adj_list_edges(size_type uid1, size_type uid2) {
		assert(uid1 < this->num_unique_nodes() && uid2 < this->num_unique_nodes() );
		//Add uids into the Adj list
		adj_list_Edges_[uid1].push_back(uid2);
		adj_list_Edges_[uid2].push_back(uid1);
		//Add an empty edge_value_type into Adj list value
		adj_list_Edges_values_[uid1].push_back(edge_value_type());
		adj_list_Edges_values_[uid2].push_back(edge_value_type());
	}

};

#endif
