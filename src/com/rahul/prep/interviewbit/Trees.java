package com.rahul.prep.interviewbit;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Stack;
import java.util.TreeMap;
import java.util.concurrent.atomic.AtomicInteger;

import javax.swing.text.AbstractDocument.LeafElement;

public class Trees {

	class ListNode {
		public int val;
		public ListNode next;

		ListNode(int x) {
			val = x;
			next = null;
		}
	}

	class TreeNode {
		int val;
		TreeNode left;
		TreeNode right;

		TreeNode(int x) {
			val = x;
			left = null;
			right = null;
		}
	}

	public ArrayList<Integer> dNums(ArrayList<Integer> A, int B) {
		ArrayList<Integer> distinctNums = new ArrayList<>();
		if (B > A.size())
			return distinctNums;

		Map<Integer, Integer> frequencyMap = new HashMap<>(B);

		// add A[0] to A[B-1] to map
		for (int i = 0; i < B; i++) {
			if (frequencyMap.containsKey(A.get(i))) {
				frequencyMap.put(A.get(i), frequencyMap.get(A.get(i)) + 1);
			} else {
				frequencyMap.put(A.get(i), 1);
			}
		}

		// add first distinct count
		distinctNums.add(frequencyMap.keySet().size());

		// iterate through A[B] to A[A.size()]
		for (int i = B; i < A.size(); i++) {
			// remove A[i-B] element from map
			if (frequencyMap.get(A.get(i - B)) == 1) {
				frequencyMap.remove(A.get(i - B));
			} else {
				frequencyMap.put(A.get(i - B), frequencyMap.get(A.get(i - B)) - 1);
			}

			// add A[i] to map
			if (frequencyMap.containsKey(A.get(i))) {
				frequencyMap.put(A.get(i), frequencyMap.get(A.get(i)) + 1);
			} else {
				frequencyMap.put(A.get(i), 1);
			}

			// get latest distinct count
			distinctNums.add(frequencyMap.keySet().size());
		}

		return distinctNums;
	}

	public ArrayList<Integer> solve(ArrayList<Integer> A, ArrayList<Integer> B) {
		ArrayList<Integer> maxPairSum = new ArrayList<>();

		// trivial case
		if (A.size() != B.size())
			return maxPairSum;
		if (A.size() == 0 && B.size() == 0)
			return maxPairSum;

		// impl
		PriorityQueue<Integer> maxPairQueue = new PriorityQueue<>(A.size());
		Collections.sort(A, (Integer i1, Integer i2) -> i2.compareTo(i1));
		Collections.sort(B, (Integer i1, Integer i2) -> i2.compareTo(i1));
		for (int a : A) {
			for (int b : B) {
				if (maxPairQueue.size() < A.size())
					maxPairQueue.add(a + b);
				else {
					if (a + b > maxPairQueue.peek()) {
						maxPairQueue.poll();
						maxPairQueue.add(a + b);
					} else
						break;
				}
			}
		}

		Iterator<Integer> iterator = maxPairQueue.iterator();
		int counter = 0;
		while (iterator.hasNext() && counter++ < A.size()) {
			maxPairSum.add(maxPairQueue.poll());
		}

		Collections.sort(maxPairSum, Collections.reverseOrder());

		return maxPairSum;
	}

	public int nchoc(int A, ArrayList<Integer> B) {
		int maxChocs = 0;
		int max = 1000000007;
		PriorityQueue<Integer> bag = new PriorityQueue<>(B.size(), Collections.reverseOrder());
		for (int numChocs : B)
			bag.add(numChocs);

		while (A-- > 0) {
			int maxChocsInBag = bag.peek();
			maxChocs += (bag.poll() % max);
			if (maxChocs >= max)
				maxChocs %= max;
			bag.add((int) Math.floor(maxChocsInBag / 2));
		}
		return maxChocs;
	}

	public ListNode mergeKLists(ArrayList<ListNode> a) {
		PriorityQueue<ListNode> maxHeap = new PriorityQueue<>(a.size(), (n1, n2) -> Integer.compare(n1.val, n2.val));
		for (ListNode node : a)
			maxHeap.add(node);
		ListNode root = maxHeap.peek();
		ListNode prev = null;
		ListNode curr = null;
		while (!maxHeap.isEmpty()) {
			curr = maxHeap.poll();
			if (prev != null)
				prev.next = curr;
			if (curr.next != null)
				maxHeap.add(curr.next);
			prev = curr;
		}
		return root;
	}
	
	public ArrayList<ArrayList<Integer>> zigzagLevelOrder(TreeNode A) {
		LinkedList<Integer>[] levelOrderElems = new LinkedList[getTreeHeight(A)];
		for(int i = 0 ; i < levelOrderElems.length ; i++)
			levelOrderElems[i] =  new LinkedList<>();
		ArrayList<ArrayList<Integer>> levelOrderElemsList = new ArrayList<>();
		traverseTree(A, 0, levelOrderElems);
		for(LinkedList<Integer> list : levelOrderElems) {
			levelOrderElemsList.add(new ArrayList<>(list));
		}
		return levelOrderElemsList;
    }
	
	private void traverseTree(TreeNode A, int level, LinkedList<Integer>[] levelOrderElems) {
		
		if(A.left == null && A.right == null)
			return;
		
//		if(levelOrderElems.length < level)
//			levelOrderElems[level] = new LinkedList<>();
		if(level % 2 == 0) {
			levelOrderElems[level].addFirst(A.val);
		} else {
			levelOrderElems[level].addLast(A.val);
		}
		
		if(A.left != null)
			traverseTree(A.left, level+1, levelOrderElems);
		if(A.right != null)
			traverseTree(A.right, level+1, levelOrderElems);
	}
	
	private int getTreeHeight(TreeNode root) {
		if(root == null)
			return 0;
		return 1 + Math.max(getTreeHeight(root.left), getTreeHeight(root.right));
	}
	
	class TreeNodeWithLevel {
		TreeNode node;
		int level;
		
		TreeNodeWithLevel(TreeNode node, int level) {
			this.node = node;
			this.level = level;
		}
	}
	
	public ArrayList<ArrayList<Integer>> verticalOrderTraversal(TreeNode A) {
		ArrayList<ArrayList<Integer>> verticalLevelList = new ArrayList<>();
	    if(A == null)
	        return verticalLevelList;
		LinkedList<TreeNodeWithLevel> queue = new LinkedList<>();
		TreeMap<Integer, ArrayList<Integer>> verticalLevelMap = new TreeMap<>();
		queue.add(new TreeNodeWithLevel(A, 0));
		while(!queue.isEmpty()) {
			TreeNodeWithLevel node = queue.removeFirst();
			int verticalLevel = node.level;
			if(verticalLevelMap.containsKey(verticalLevel))
				verticalLevelMap.get(verticalLevel).add(node.node.val);
			else {
				ArrayList<Integer> verticalOrderList = new ArrayList<>();
				verticalOrderList.add(node.node.val);
				verticalLevelMap.put(verticalLevel, verticalOrderList);
			}
			if(node.node.left != null)
				queue.addLast(new TreeNodeWithLevel(node.node.left, node.level - 1));
			if(node.node.right != null)
				queue.addLast(new TreeNodeWithLevel(node.node.right, node.level + 1));
		}
		for(Integer key : verticalLevelMap.keySet())
			verticalLevelList.add(verticalLevelMap.get(key));
		return verticalLevelList;
    }
	
	private static final int MAX_ALPHABET = 26;
	
	class TrieNode {
		char val;
		boolean isEndOfWord;
		TrieNode children[];
		int index;
		
		TrieNode(char val) {
			this.val = val;
			this.isEndOfWord = false;
			this.children = new TrieNode[MAX_ALPHABET];
			this.index = -1;
		}
	}
	
	private int charToIndex(char c) {
		return c - 97;
	}
	
	private boolean hasChildren(TrieNode node) {
		for(int i = 0 ; i < MAX_ALPHABET ; i++)
			if(node.children[i] != null)
				return true;
		
		return false;
	}
	
	private void insert(TrieNode node, String word, int index) {
		for(int i = 0 ; i < word.length() ; i++) {
			if(node.children[charToIndex(word.charAt(i))] == null)
				node.children[charToIndex(word.charAt(i))] = new TrieNode(word.charAt(i));
			node = node.children[charToIndex(word.charAt(i))];
		}
		node.isEndOfWord = true;
		node.index = index;
	}

	public ArrayList<String> shortestUniquePrefix(ArrayList<String> A) {
		TrieNode root = new TrieNode('\0');
		AtomicInteger counter = new AtomicInteger(0);
		A.forEach(s -> {insert(root, s, counter.getAndIncrement());});
		ArrayList<String> prefixList = new ArrayList<>();
		TreeMap<Integer, String> prefixMap = getShortestUniquePrefixOfChild(root);
//		for (Map.Entry<Integer, String> entry : prefixMap.entrySet()) {
//		     System.out.println("Key: " + entry.getKey() + ". Value: " + entry.getValue());
//		}
		prefixList.addAll(prefixMap.values());
		return prefixList;
    }
	
	private TreeMap<Integer, String> getShortestUniquePrefixOfChild(TrieNode node) {
		TreeMap<Integer, String> masterPrefixList = new TreeMap<>();
		if(node.isEndOfWord) {
			masterPrefixList.put(node.index, String.valueOf(node.val));
		}
		if(!hasChildren(node)) {
			return masterPrefixList;
		}
		for(int i = 0 ; i < MAX_ALPHABET ; i++) {
			if(node.children[i] != null) {
				TreeMap<Integer, String> prefixList = getShortestUniquePrefixOfChild(node.children[i]);
				masterPrefixList.putAll(prefixList);
			}
		}
		if(masterPrefixList.keySet().size() == 1) {
			masterPrefixList.put(masterPrefixList.firstKey(), String.valueOf(node.val));
		} else {
			if(node.val != '\0')
				for(Integer index : masterPrefixList.keySet())
					masterPrefixList.put(index, node.val + masterPrefixList.get(index));
		}
		return masterPrefixList;
	}
	
	public ArrayList<Integer> solveHotelReviews(String A, ArrayList<String> B) {
		
		//construct trie
		TrieNode root = new TrieNode('\0');
		String[] goodWords = A.split("_");
		for(int i = 0 ; i < goodWords.length ; i++)
			insert(root, goodWords[i], i);
		
		//sort reviews
		TreeMap<Integer, ArrayList<Integer>> goodnessToIndexMap = new TreeMap<>(Collections.reverseOrder());
		for(int i = 0 ; i < B.size() ; i++) {
			String[] reviewWords = B.get(i).split("_");
			int reviewGoodness = 0;
			for(String word : reviewWords) {
				reviewGoodness += isGoodWord(root, word);
			}
			if(goodnessToIndexMap.keySet().contains(reviewGoodness)) {
				goodnessToIndexMap.get(reviewGoodness).add(i);
			} else {
				ArrayList<Integer> indexList = new ArrayList<>();
				indexList.add(i);
				goodnessToIndexMap.put(reviewGoodness, indexList);
			}
		}
		
		//get list from map
		ArrayList<Integer> sortedReviewIndex = new ArrayList<>();
		for(Integer goodness : goodnessToIndexMap.keySet())
			sortedReviewIndex.addAll(goodnessToIndexMap.get(goodness));
		
		return sortedReviewIndex;
	}
	
	private int isGoodWord(TrieNode root, String word) {
		for(int i = 0 ; i < word.length() ; i++) {
			char c = word.charAt(i);
			if(root.children[charToIndex(c)] != null) {
				root = root.children[charToIndex(c)];
			} else
				return 0;
		}
		
		if(root.isEndOfWord)
			return 1;
		
		return 0;
	}
	
	class TreeLinkNode {
		int val;
		TreeLinkNode left, right, next;

		TreeLinkNode(int x) {
			val = x;
		}
	}

	public void connectAdjacentNodesInTree(TreeLinkNode root) {
		LinkedList<TreeLinkNode> queue1 = new LinkedList<>();
		LinkedList<TreeLinkNode> queue2 = new LinkedList<>();
		
		queue1.add(root);
		while(!queue1.isEmpty() || !queue2.isEmpty()) {
			if(queue2.isEmpty()) {
				TreeLinkNode prev = null;
				while(!queue1.isEmpty()) {
					TreeLinkNode curr = queue1.pollFirst();
					
					//put children inn q2
					if(curr.left != null)
						queue2.addLast(curr.left);
					if(curr.right != null)
						queue2.addLast(curr.right);
					
					if(prev != null)
						prev.next = curr;
					if(queue1.isEmpty())
						curr.next = null;
					prev = curr;
				}
			}
			if(queue1.isEmpty()) {
				TreeLinkNode prev = null;
				while(!queue2.isEmpty()) {
					TreeLinkNode curr = queue2.pollFirst();
					
					//put children inn q1
					if(curr.left != null)
						queue1.addLast(curr.left);
					if(curr.right != null)
						queue1.addLast(curr.right);
					
					if(prev != null)
						prev.next = curr;
					if(queue1.isEmpty())
						curr.next = null;
					prev = curr;
				}
			}
		}
		
	}
	
    TreeNode firstElement = null;
    TreeNode secondElement = null;
    TreeNode prevElement = new TreeNode(Integer.MIN_VALUE);
	public ArrayList<Integer> recoverTree(TreeNode A) {
		traverse(A);
		ArrayList<Integer> swaps = new ArrayList<>();
		swaps.add(firstElement.val);
		swaps.add(secondElement.val);
		return swaps;
    }
	
	private void traverse(TreeNode root) {

		if (root == null)
			return;

		traverse(root.left);

		if (firstElement == null && prevElement.val >= root.val) {
			firstElement = prevElement;
		}
		if (firstElement != null && prevElement.val >= root.val) {
			secondElement = root;
		}
		prevElement = root;
		traverse(root.right);
	}
	
	public int t2Sum(TreeNode A, int B) {
		
		TreeNode incCurr = A;
		TreeNode decCurr = A;
		Stack<TreeNode> incStack = new Stack<>();
		Stack<TreeNode> decStack = new Stack<>();
		boolean shouldAscendIncStack = true;
		boolean shouldDescendDecStack = true;
		boolean done = false;
		
		while(!done) {
			
			while(incCurr != null && shouldAscendIncStack) {
				incStack.push(incCurr);
				incCurr = incCurr.left;
			}
			
			while(decCurr != null && shouldDescendDecStack) {
				decStack.push(decCurr);
				decCurr = decCurr.right;
			}
			
			//do something
			incCurr = incStack.pop();
			decCurr = decStack.pop();
			System.out.println(incCurr.val + " " + decCurr.val );
			if(incCurr.val + decCurr.val == B && incCurr.val != decCurr.val)
				return 1;
			else if(incCurr.val + decCurr.val < B) {
				//increment incCurr pointer to next value
				incCurr = incCurr.right;
				//restore desCurr in desStack
				decStack.push(decCurr);
				//make sure descStack does not move backward
				shouldDescendDecStack = false;
				shouldAscendIncStack = true;
			} else {
				//decrement decCurr pointer to next value
				decCurr = decCurr.left;
				//restore incCurr in incStack
				incStack.push(incCurr);
				//make sure incStack does not move forward
				shouldAscendIncStack = false;
				shouldDescendDecStack = true;
			}
			
			if((incCurr == null && incStack.isEmpty()) || (decCurr == null && decStack.isEmpty()))
				done = true;

		}
		
		return 0;
    }
	
	public int lcaForBST(TreeNode root, int val1, int val2) {
		
		LinkedList<TreeNode> queue1 = new LinkedList<>();
		LinkedList<TreeNode> queue2 = new LinkedList<>();
		boolean foundVal1 = false;
		boolean foundVal2 = false;
		
		TreeNode curr = root;
		while(curr != null) {
			queue1.addFirst(curr);
			if(curr.val == val1) {
				foundVal1 = true;
				break;
			}
			else if(curr.val < val1)
				curr = curr.right;
			else
				curr = curr.left;
		}
		curr = root;
		while(curr != null) {
			queue2.addFirst(curr);
			if(curr.val == val2) {
				foundVal2 = true;
				break;
			}
			else if(curr.val < val2)
				curr = curr.right;
			else
				curr = curr.left;
		}
		if(!foundVal1 || !foundVal2)
			return -1;
		
		TreeNode qLast1 = null;
		TreeNode qLast2 = null;
		while (!queue1.isEmpty() && !queue2.isEmpty() && queue1.peekLast() == queue2.peekLast()) {
			qLast1 = queue1.removeLast();
			qLast2 = queue2.removeLast();
			System.out.println(qLast1 + " " + qLast2);
		}
		return qLast1.val;
    }
	
	public int lca(TreeNode root, int val1, int val2) {

		LinkedList<TreeNode> queue1 = new LinkedList<>();
		LinkedList<TreeNode> queue2 = new LinkedList<>();
		if (!findElementInTree(queue1, root, val1) && !findElementInTree(queue2, root, val2))
			return -1;

		Iterator<TreeNode> it1 = queue1.iterator();
		Iterator<TreeNode> it2 = queue2.iterator();
		while (it1.hasNext())
			System.out.println(it1.next().val);
		System.out.println();
		while (it2.hasNext())
			System.out.println(it2.next().val);

		TreeNode qLast1 = null;
		TreeNode qLast2 = null;
		while (!queue1.isEmpty() && !queue2.isEmpty() && queue1.peekLast() == queue2.peekLast()) {
			qLast1 = queue1.removeLast();
			qLast2 = queue2.removeLast();
			System.out.println(qLast1 + " " + qLast2);
		}
		return qLast1.val;
	}

	private boolean findElementInTree(LinkedList<TreeNode> queue, TreeNode node, int val) {
		if(node == null)
			return false;
		queue.addFirst(node);
		if(node.val == val) {
			return true;
		}
		boolean foundInLeftSubtree = findElementInTree(queue, node.left, val);
		boolean foundInRightSubtree = findElementInTree(queue, node.right, val);
		
		if(!foundInLeftSubtree && !foundInRightSubtree) {
			queue.pop();
		}
		
		return foundInLeftSubtree || foundInRightSubtree;
	}
	
	//NOT WORKING
	public TreeNode buildTreeFromPostorderAndInorder(ArrayList<Integer> inorder,
			ArrayList<Integer> postorder) {
		
		return buildTreeFromPostorderAndInorderRec(inorder, 0, inorder.size() - 1, postorder, 0, postorder.size() - 1);
	}
	private TreeNode buildTreeFromPostorderAndInorderRec(ArrayList<Integer> inorder, int inorderStart, int inorderEnd,
			ArrayList<Integer> postorder, int postorderStart, int postorderEnd) {

		if(inorderStart > inorderEnd)
			return null;
		
		if(inorderStart == inorderEnd)
			return new TreeNode(inorder.get(inorderStart));
		
		TreeNode root = new TreeNode(postorder.get(postorderEnd));
		
		int indexOfRootInInorder = -1;
		for(int i = inorderStart ; i <= inorderEnd ; i++) {
			if(inorder.get(i) == root.val)
				indexOfRootInInorder = i;
		}
		int indexOfLeftMostElementInRightSubtree = -1;
		if(indexOfRootInInorder + 1 < inorder.size()) {
			for(int i = postorderStart ; i <= postorderEnd ; i++) {
				if(postorder.get(i) == inorder.get(indexOfRootInInorder + 1))
					indexOfLeftMostElementInRightSubtree = i;
			}
		} else {
			
		}
		root.left = buildTreeFromPostorderAndInorderRec(inorder, inorderStart, indexOfRootInInorder - 1, postorder, postorderStart, indexOfLeftMostElementInRightSubtree - 1 );
		root.right = buildTreeFromPostorderAndInorderRec(inorder, indexOfRootInInorder + 1, inorderEnd, postorder, indexOfLeftMostElementInRightSubtree, postorderEnd );
		
		return root;
	}
	
	public TreeNode buildCartesianTree(ArrayList<Integer> A) {
		return buildCartesianTreeRec(A, 0, A.size() - 1);
    }
	
	private TreeNode buildCartesianTreeRec(ArrayList<Integer> A, int start, int end) {
		
		//base case
		if(start > end)
			return null;
		
		//find max
		int max = Integer.MIN_VALUE;
		int rootIndex = -1;
		for(int i = start; i <= end; i++) {
			if(max < A.get(i)) {
				max = A.get(i);
				rootIndex = i;
			}
		}
		
		TreeNode root = new TreeNode(max);
		root.left = buildCartesianTreeRec(A, start, rootIndex - 1);
		root.right = buildCartesianTreeRec(A, rootIndex + 1, end);
		
		return root;
	}
	
	ArrayList<ArrayList<Integer>> pathList = new ArrayList<>();
	public ArrayList<ArrayList<Integer>> pathSum(TreeNode A, int B) {
		tranverseToLeaf(new ArrayList<Integer>(), A, B);
		return pathList;
    }
	
	private void tranverseToLeaf(ArrayList<Integer> currentpath, TreeNode node, int sum) {
		
		//base case
		if(node == null)
			return;
		if(node.left == null && node.right == null && sum == node.val) {
			ArrayList<Integer> path = new ArrayList<>(currentpath);
			path.add(node.val);
			return;
		}
		if(node.left == null && node.right == null && sum != node.val)
			return;
		
		currentpath.add(node.val);
		tranverseToLeaf(currentpath, node.left, sum - node.val);
		tranverseToLeaf(currentpath, node.right, sum - node.val);
		currentpath.remove(currentpath.size() - 1);
		
	}
	
	public static void main(String[] args) {
		Trees soln = new Trees();
		ArrayList<String> inp = new ArrayList<>();
		inp.add("zebra");
		inp.add("dog");
		inp.add("duck");
		inp.add("dot");
		ArrayList<String> out = soln.shortestUniquePrefix(inp);
		String t = '\0' + "";
		System.out.println(t.equals(""));
		out.forEach(s -> System.out.println(s));
		TreeNode root = soln.new TreeNode(10);
		TreeNode c1 = soln.new TreeNode(9);
		TreeNode c2 = soln.new TreeNode(20);
		root.left = c1;
		root.right = c2;
		System.out.println(soln.t2Sum(root, 40));
	}

}
