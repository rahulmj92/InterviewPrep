package com.rahul.prep.interviewbit;

import java.math.BigDecimal;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Stack;

public class DynamicProgramming {

	/*
	 * Given an array of integers, find the length of longest subsequence which
	 * is first increasing then decreasing.
	 */
	public int longestSubsequenceLength(final List<Integer> A) {

		// init
		int lis[] = new int[A.size()];
		int lds[] = new int[A.size()];
		Arrays.fill(lis, 1);
		Arrays.fill(lds, 1);

		// find lis in a bottom-up approach
		for (int i = 0; i < A.size(); i++) {
			for (int j = 0; j < i; j++) {
				if (A.get(j) < A.get(i) && lis[i] < lis[j] + 1) {
					lis[i] = lis[i] + 1;
				}
			}
		}

		// find lds in a bottom-up approach
		for (int i = A.size() - 1; i >= 0; i--) {
			for (int j = A.size() - 1; j > i; j--) {
				if (A.get(j) < A.get(i) && lds[i] < lds[j] + 1) {
					lds[i] = lds[i] + 1;
				}
			}
		}

		int longestSubsequenceLength = 0;
		for (int i = 0; i < A.size(); i++) {
			if (longestSubsequenceLength < lis[i] + lds[i] - 1)
				longestSubsequenceLength = lis[i] + lds[i] - 1;
		}
		return longestSubsequenceLength;
	}

	public int climbStairs(int A) {

		if (A == 0)
			return 0;

		if (A == 1)
			return 1;

		if (A == 2)
			return 2;

		int uniqueSteps[] = new int[A];

		uniqueSteps[0] = 1;
		uniqueSteps[1] = 2;
		for (int i = 2; i < A; i++) {
			uniqueSteps[i] = uniqueSteps[i - 1] + uniqueSteps[i - 2];
		}

		return uniqueSteps[A - 1];
	}

	public int numDecodings(String A) {

		if (A == null)
			return 0;

		if (A.length() == 0)
			return 0;

		if (A.length() == 1) {
			if (getIntegerValue(A.charAt(0)) != 0)
				return 1;
			else
				return 0;
		}

		int numDecodings[] = new int[A.length()];
		// for cases 01, 02
		if (getIntegerValue(A.charAt(0)) == 0) {
			numDecodings[0] = 0;
			numDecodings[1] = 0;
		} else if (getIntegerValue(A.charAt(0)) * 10 + getIntegerValue(A.charAt(1)) > 26) {
			numDecodings[0] = 1;
			numDecodings[1] = 1;
		} else {
			// for cases 10 and 30
			if (getIntegerValue(A.charAt(1)) != 0) {
				numDecodings[0] = 1;
				numDecodings[1] = 2;
			} else {
				numDecodings[0] = 1;
				numDecodings[1] = 1;
			}
		}

		for (int i = 2; i < A.length(); i++) {
			// System.out.println((getIntegerValue(A.charAt(i - 1)) * 10) +
			// getIntegerValue(A.charAt(i)));
			if (getIntegerValue(A.charAt(i - 1)) != 0
					&& (getIntegerValue(A.charAt(i - 1)) * 10) + getIntegerValue(A.charAt(i)) <= 26) {
				// cases involving A[i-1] = 1/2 and A[i] = 0
				if (getIntegerValue(A.charAt(i)) == 0)
					numDecodings[i] = numDecodings[i - 2];
				else
					numDecodings[i] = numDecodings[i - 1] + numDecodings[i - 2];
			} else if (getIntegerValue(A.charAt(i)) != 0)
				numDecodings[i] = numDecodings[i - 1];
			else
				numDecodings[i] = 0;
			System.out.println("i : " + (i + 1) + " val : " + numDecodings[i]);
		}

		return numDecodings[A.length() - 1];
	}

	private int getIntegerValue(char c) {
		return c - 48;
	}

	public int solveLargestAreaOfRectWithPermutations(ArrayList<ArrayList<Integer>> A) {

		if (A == null)
			return 0;

		if (A.size() == 0)
			return 0;

		int numRows = A.size();
		int numCols = A.get(0).size();
		for (int col = 0; col < numCols; col++) {
			for (int row = 1; row < numRows; row++) {
				if (A.get(row - 1).get(col) > 0 && A.get(row).get(col) != 0)
					A.get(row).set(col, A.get(row - 1).get(col) + 1);
			}
		}

		int max = Integer.MIN_VALUE;
		for (int row = 0; row < numRows; row++) {
			int count[] = new int[numRows + 1];
			int length = 0;
			Arrays.fill(count, 0);
			for (int col = 0; col < numCols; col++)
				count[A.get(row).get(col)]++;
			for (int i = numRows; i >= 0; i--) {
				length += count[i];
				max = Math.max(max, length * i);
			}
		}

		return max;
	}

	// Ways(n) = sum[i = 0 to n-1] { Ways(i)*Ways(n-i-1) }.
	public int chordCnt(int A) {
		chordCntMem = new int[2 * A + 1];
		Arrays.fill(chordCntMem, -1);
		return chordCntRec(A * 2);

	}

	private static final int MAX = 1000000007;
	private static int chordCntMem[];

	private int chordCntRec(int A) {

		if (chordCntMem[A] != -1)
			return chordCntMem[A];

		if (A == 0)
			return 1;

		if (A == 1)
			return 0;

		if (A == 2)
			return 1;

		int cnt = 0;
		for (int i = 0; i <= A - 2; i++) {
			int firstSetSize = i;
			int secondSetSize = A - 2 - i;
			BigDecimal holder1 = new BigDecimal(chordCntRec(firstSetSize) % MAX);
			BigDecimal holder2 = new BigDecimal(chordCntRec(secondSetSize) % MAX);
			cnt += holder1.multiply(holder2).divideAndRemainder(new BigDecimal(MAX))[1].intValue();
			if (cnt >= MAX)
				cnt %= MAX;
			// System.out.println(cnt);
		}
		// memoization
		chordCntMem[A] = cnt;
		return cnt;
	}

	public int maxcoin(ArrayList<Integer> A) {
		int C[][] = new int[A.size()][A.size()];
		for (int i = A.size() - 1; i >= 0; i--) {
			for (int j = A.size() - 1; j >= 0; j--) {
				if (i <= j) {
					if (i == j)
						C[i][j] = A.get(i);
					else if (i == j + 1)
						C[i][j] = Math.max(A.get(i), A.get(j));
					else
						C[i][j] = Math.max(A.get(i) + Math.min(C[i + 1][j - 1], C[i + 2][j]),
								A.get(j) + Math.min(C[i - 1][j - 1], C[i][j - 2]));
				}
			}
		}
		return C[0][A.size() - 1];
	}

	// Longest Arithmetic Progression
	public int solve(final List<Integer> A) {
		Collections.sort(A);
		// int set[] = new int[A.size()];
		int n = A.size();

		if (n <= 2)
			return n;

		// Create a table and initialize all
		// values as 2. The value ofL[i][j] stores
		// LLAP with set[i] and set[j] as first two
		// elements of AP. Only valid entries are
		// the entries where j>i
		int L[][] = new int[n][n];

		// Initialize the result
		int llap = 2;

		// Fill entries in last column as 2.
		// There will always be two elements in
		// AP with last number of set as second
		// element in AP
		for (int i = 0; i < n; i++)
			L[i][n - 1] = 2;

		// Consider every element as second element of AP
		for (int j = n - 2; j >= 1; j--) {
			// Search for i and k for j
			int i = j - 1, k = j + 1;
			while (i >= 0 && k <= n - 1) {
				if (A.get(i) + A.get(k) < 2 * A.get(j))
					k++;

				// Before changing i, set L[i][j] as 2
				else if (A.get(i) + A.get(k) > 2 * A.get(j)) {
					L[i][j] = 2;
					i--;

				}

				else {
					// Found i and k for j, LLAP with i and j as first two
					// elements is equal to LLAP with j and k as first two
					// elements plus 1. L[j][k] must have been filled
					// before as we run the loop from right side
					L[i][j] = L[j][k] + 1;

					// Update overall LLAP, if needed
					llap = Math.max(llap, L[i][j]);

					// Change i and k to fill
					// more L[i][j] values for current j
					i--;
					k++;
				}
			}

			// If the loop was stopped due
			// to k becoming more than
			// n-1, set the remaining
			// entities in column j as 2
			while (i >= 0) {
				L[i][j] = 2;
				i--;
			}
		}
		return llap;
	}

	public static int numTreeMem[];

	public int numTrees(int A) {
		numTreeMem = new int[A + 1];
		Arrays.fill(numTreeMem, 0);
		return numTreesRec(A);
	}

	private int numTreesRec(int A) {

		if (numTreeMem[A] != 0)
			return numTreeMem[A];

		if (A <= 1)
			return 1;

		int numTrees = 0;
		// choose one element X from 1..N,
		// and divide left and right elements of X into 2 sets
		for (int i = 1; i <= A; i++) {
			int setSize1 = i - 1;
			int setSize2 = A - i;
			numTrees += numTreesRec(setSize1) * numTreesRec(setSize2);
		}
		numTreeMem[A] = numTrees;
		return numTrees;

	}

	public int longestValidParentheses(String A) {

		int lvp[] = new int[A.length()];
		Stack<Character> st = new Stack<>();
		for (int i = 0; i < A.length(); i++) {
			if (A.charAt(i) == '(') {
				if (i == 0) {
					lvp[i] = 0;
					continue;
				}
				lvp[i] = lvp[i - 1];
				st.push(A.charAt(i));
			} else {
				if (st.empty()) {
					lvp[i] = 0;
				} else if (st.peek() == '(') {
					lvp[i] = lvp[i - 1] + 2;
					st.pop();
				}
			}
			System.out.println("st len: " + st.size() + " " + A.charAt(i) + " " + lvp[i]);
		}
		int max = Integer.MIN_VALUE;
		for (int i = 0; i < lvp.length; i++)
			if (max < lvp[i])
				max = lvp[i];

		return max;
	}

	public int longestValidParenthesesusingStack(String s) {

		int maxans = 0;
		Stack<Integer> stack = new Stack<>();
		stack.push(-1);
		for (int i = 0; i < s.length(); i++) {
			if (s.charAt(i) == '(') {
				stack.push(i);
			} else {
				stack.pop();
				if (stack.empty()) {
					stack.push(i);
				} else {
					maxans = Math.max(maxans, i - stack.peek());
				}
			}
		}
		return maxans;
	}

	public int maxProfit(final List<Integer> A) {

		if (A == null || A.size() == 0 || A.size() == 1)
			return 0;

		int minSoFar = A.get(0);
		int maxProfit = 0;
		int minIndex = -1;

		for (int i = 1; i < A.size(); i++) {
			if (minSoFar > A.get(i)) {
				minSoFar = A.get(i);
				minIndex = i;
			}
			if (A.get(i) - minSoFar > maxProfit && i > minIndex)
				maxProfit = A.get(i) - minSoFar;
		}

		return maxProfit;
	}

	public static int coinChangeMem[][];

	public int coinchange2(ArrayList<Integer> coins, int sum) {

		coinChangeMem = new int[coins.size()][sum + 1];
		for (int row = 0; row < coins.size(); row++)
			Arrays.fill(coinChangeMem[row], -1);

		return coinChange2Rec(coins, sum, 0);
	}

	private int coinChange2Rec(ArrayList<Integer> coins, int sum, int index) {

		// System.out.println(coinChangeMem[index][sum]);

		if (sum < 0)
			return 0;

		if (sum == 0)
			return 1;

		if (coinChangeMem[index][sum] != -1)
			return coinChangeMem[index][sum];

		int count = 0;
		for (int i = index; i < coins.size(); i++) {
			count += (coinChange2Rec(coins, sum - coins.get(i), i) % 1000007);
			System.out.println(count);
			if (count >= 1000007)
				count %= 1000007;
		}

		coinChangeMem[index][sum] = count;
		return count;
	}

	public int coinchange2New(ArrayList<Integer> coins, int sum) {

		return 0;
	}

	// Evaluate Expression To True
	public int cnttrue(String A) {

		if (A == null)
			return 0;

		int numOperators = (A.length() - 1) / 2;
		int trueWays[][] = new int[numOperators][numOperators];
		int falseWays[][] = new int[numOperators][numOperators];

		for (int i = 0; i < numOperators; i++) {
			Arrays.fill(trueWays[i], 0);
			Arrays.fill(falseWays[i], 0);
		}

		for (int i = 0; i < numOperators; i++) {
			char operand1 = A.charAt(i * 2);
			char operand2 = A.charAt(i * 2 + 2);
			char operator = A.charAt(i * 2 + 1);
			if (operator == '|') {
				trueWays[i][i] = ((operand1 == 'T' ? true : false) | (operand2 == 'T' ? true : false) ? 1 : 0);
				falseWays[i][i] = ((operand1 == 'T' ? true : false) | (operand2 == 'T' ? true : false) ? 0 : 1);
			} else if (operator == '&') {
				trueWays[i][i] = ((operand1 == 'T' ? true : false) & (operand2 == 'T' ? true : false) ? 1 : 0);
				falseWays[i][i] = ((operand1 == 'T' ? true : false) & (operand2 == 'T' ? true : false) ? 0 : 1);
			} else if (operator == '^') {
				trueWays[i][i] = ((operand1 == 'T' ? true : false) ^ (operand2 == 'T' ? true : false) ? 1 : 0);
				falseWays[i][i] = ((operand1 == 'T' ? true : false) ^ (operand2 == 'T' ? true : false) ? 0 : 1);
			}
			System.out.println(trueWays[i][i] + " " + falseWays[i][i]);
		}

		for (int i = numOperators - 1; i >= 0; i--) {
			for (int j = i; j < numOperators; j++) {
				for (int k = i; k < j; k++) {
					char operator = A.charAt(k * 2 + 1);
					if (operator == '&') {
						trueWays[i][j] += trueWays[i][k] * trueWays[k + 1][j];
						falseWays[i][j] += falseWays[i][k] * falseWays[k + 1][j] + trueWays[i][k] * falseWays[k + 1][j]
								+ falseWays[i][k] * trueWays[k + 1][j];
					} else if (operator == '|') {
						trueWays[i][j] += trueWays[i][k] * trueWays[k + 1][j] + trueWays[i][k] * falseWays[k + 1][j]
								+ falseWays[i][k] * trueWays[k + 1][j];
						falseWays[i][j] += falseWays[i][k] * falseWays[k + 1][j];
					} else if (operator == '^') {
						trueWays[i][j] += trueWays[i][k] * falseWays[k + 1][j] + falseWays[i][k] * trueWays[k + 1][j];
						falseWays[i][j] += trueWays[i][k] * trueWays[k + 1][j] + falseWays[i][k] * falseWays[k + 1][j];
					}
				}
			}
		}
		return trueWays[0][numOperators - 1];
	}

	public int solveNdigitSsum(int n, int sum) {

		// base cases
		if (n <= 0 || sum <= 0)
			return 0;
		if (n == 1 && sum > 10)
			return 0;
		if (n == 1 && sum < 10)
			return 1;

		// memoization
		int count[][] = new int[n + 1][sum + 1];

		// init memoization base cases
		for (int i = 0; i <= n; i++)
			Arrays.fill(count[i], 0);

		for (int s = 0; s <= sum; s++) {
			if (s < 10)
				count[1][s] = 1;
			else
				count[1][s] = 0;
		}
		for (int d = 0; d <= n; d++)
			count[d][0] = 1;

		// bottom-up approach
		for (int numDigit = 2; numDigit <= n; numDigit++) {
			for (int remSum = 1; remSum <= sum; remSum++) {
				for (int digit = 0; digit <= 9; digit++) {
					// removing cases where first digit cannot be zero
					if (!(digit == 0 && numDigit == n))
						if (remSum - digit >= 0)
							count[numDigit][remSum] += count[numDigit - 1][remSum - digit];
				}
				System.out.println("count[" + numDigit + "][" + remSum + "]: " + count[numDigit][remSum]);
			}
		}

		return count[n][sum];
	}

	// Edit Distance
	public int minDistance(String A, String B) {

		if (A == null || B == null)
			return 0;

		if (A.length() == 0)
			return B.length();

		if (B.length() == 0)
			return A.length();

		int m = A.length();
		int n = B.length();

		int dist[][] = new int[m][n];

		// init mem array
		for (int i = 0; i < m; i++)
			Arrays.fill(dist[i], 0);

		// init base cases of mem
		for (int i = 1; i < m; i++)
			dist[i][0] = i + 1;

		for (int i = 1; i < n; i++)
			dist[0][i] = i + 1;

		// bottom-up approach
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (i == 0)
					dist[i][j] = j + 1;
				if (j == 0)
					dist[i][j] = i + 1;

				if (A.charAt(i) == B.charAt(j))
					dist[i][j] = dist[i - 1][j - 1];
				else {
					dist[i][j] = 1 + Math.min(dist[i - 1][j - 1], Math.min(dist[i][j - 1], dist[i - 1][j]));
				}
			}
		}

		return dist[m][n];
	}

	public int lis(final List<Integer> A) {
		// base cases

		int lis[] = new int[A.size()];
		Arrays.fill(lis, 1);
		for (int i = 0; i < A.size(); i++) {
			for (int j = 0; j < i; j++) {
				if (A.get(i) > A.get(j) && lis[i] < lis[j] + 1)
					lis[i] = lis[j] + 1;
			}
		}
		int llis = Integer.MIN_VALUE;
		for (int i = 0; i < A.size(); i++) {
			llis = Math.max(llis, lis[i]);
		}
		return llis;
	}

	public int isMatch(final String str, final String regex) {

		int strLen = str.length();
		int regexLen = regex.length();
		boolean match[][] = new boolean[strLen + 1][regexLen + 1];

		// init array
		for (int i = 1; i <= strLen; i++)
			Arrays.fill(match[i], false);
		match[0][0] = true;
		for (int i = 1; i <= strLen; i++)
			match[i][0] = false;
		for (int j = 1; j <= regexLen; j++)
			if (regex.charAt(j - 1) == '*')
				match[0][j] = match[0][j - 1];

		// bottom-up solve
		for (int i = 1; i <= strLen; i++) {
			for (int j = 1; j <= regexLen; j++) {
				if (regex.charAt(j - 1) == '*') {
					match[i][j] = match[i - 1][j] || match[i][j - 1];
				} else if (regex.charAt(j - 1) == '?' || str.charAt(i - 1) == regex.charAt(j - 1)) {
					match[i][j] = match[i - 1][j - 1];
				} else {
					match[i][j] = false;
				}
			}
		}

		return match[strLen][regexLen] ? 1 : 0;
	}

	public int isMatch2(final String str, final String regex) {

		int strLen = str.length();
		int regexLen = regex.length();
		boolean match[][] = new boolean[strLen + 1][regexLen + 1];

		// init array
		for (int i = 1; i <= strLen; i++)
			Arrays.fill(match[i], false);
		match[0][0] = true;
		for (int i = 1; i <= strLen; i++)
			match[i][0] = false;
		for (int j = 1; j <= regexLen; j++)
			if (regex.charAt(j - 1) == '*')
				match[0][j] = match[0][j - 2];

		// bottom-up solve
		for (int i = 1; i <= strLen; i++) {
			for (int j = 1; j <= regexLen; j++) {
				if (regex.charAt(j - 1) == '*') {
					if (str.charAt(i - 1) == regex.charAt(j - 2) || regex.charAt(j - 2) == '.')
						match[i][j] = match[i][j - 2] || match[i - 1][j];
					else
						match[i][j] = match[i][j - 2];
				} else if (regex.charAt(j - 1) == '.' || str.charAt(i - 1) == regex.charAt(j - 1)) {
					match[i][j] = match[i - 1][j - 1];
				} else {
					match[i][j] = false;
				}
			}
		}

		return match[strLen][regexLen] ? 1 : 0;
	}

	private static int[][][][] scrambleMem;

	public int isScramble(final String original, final String scrambled) {

		scrambleMem = new int[original.length()][original.length()][original.length()][original.length()];
		for (int i = 0; i < original.length(); i++)
			for (int j = 0; j < original.length(); j++)
				for (int k = 0; k < original.length(); k++)
					Arrays.fill(scrambleMem[i][j][k], -1);
		return isScrambleRecHelper(original, scrambled, 0, original.length() - 1, 0, scrambled.length() - 1) ? 1 : 0;

	}

	private boolean isScrambleRecHelper(final String original, final String scrambled, int orgStartIndex,
			int orgEndIndex, int scrStartIndex, int scrEndIndex) {

		System.out.println(orgStartIndex + " " + orgEndIndex + " | " + scrStartIndex + " " + scrEndIndex);

		if (scrambleMem[orgStartIndex][orgEndIndex][scrStartIndex][scrEndIndex] != -1) {
			System.out.println("In mem");
			return scrambleMem[orgStartIndex][orgEndIndex][scrStartIndex][scrEndIndex] == 0 ? false : true;
		}

		// base cases
		if (orgEndIndex == orgStartIndex && scrStartIndex == scrEndIndex) {
			if (original.charAt(orgStartIndex) == scrambled.charAt(scrStartIndex)) {
				scrambleMem[orgStartIndex][orgEndIndex][scrStartIndex][scrEndIndex] = 1;
				return true;
			} else {
				scrambleMem[orgStartIndex][orgEndIndex][scrStartIndex][scrEndIndex] = 0;
				return false;
			}
		}

		if (!isAnagram(original, scrambled, orgStartIndex, orgEndIndex, scrStartIndex, scrEndIndex)) {
			System.out.println("not anagram");
			scrambleMem[orgStartIndex][orgEndIndex][scrStartIndex][scrEndIndex] = 0;
			return false;
		}

		for (int i = 0; i < orgEndIndex - orgStartIndex; i++) {
			if (isScrambleRecHelper(original, scrambled, orgStartIndex, orgStartIndex + i, scrStartIndex,
					scrStartIndex + i)
					&& isScrambleRecHelper(original, scrambled, orgStartIndex + i + 1, orgEndIndex,
							scrStartIndex + i + 1, scrEndIndex)) {
				scrambleMem[orgStartIndex][orgEndIndex][scrStartIndex][scrEndIndex] = 1;
				return true;
			} else if (isScrambleRecHelper(original, scrambled, orgStartIndex, orgStartIndex + i, scrEndIndex - i,
					scrEndIndex)
					&& isScrambleRecHelper(original, scrambled, orgStartIndex + i + 1, orgEndIndex, scrStartIndex,
							scrEndIndex - i - 1)) {
				scrambleMem[orgStartIndex][orgEndIndex][scrStartIndex][scrEndIndex] = 1;
				return true;
			}
		}
		scrambleMem[orgStartIndex][orgEndIndex][scrStartIndex][scrEndIndex] = 0;
		return false;

	}

	private boolean isAnagram(String org, String scr, int orgStartIndex, int orgEndIndex, int scrStartIndex,
			int scrEndIndex) {
		int orgCharset[] = new int[26];
		Arrays.fill(orgCharset, 0);
		for (int i = orgStartIndex; i <= orgEndIndex; i++)
			orgCharset[org.toLowerCase().charAt(i) - 97]++;
		for (int i = scrStartIndex; i <= scrEndIndex; i++)
			orgCharset[scr.toLowerCase().charAt(i) - 97]--;

		for (int i = 0; i < 26; i++)
			if (orgCharset[i] != 0)
				return false;
		return true;
	}

	public int repeatingSubsequence(String A) {

		if (A == null || A.length() == 0)
			return 0;

		int n = A.length();
		int rs[][] = new int[n + 1][n + 1];

		for (int i = 0; i <= n; i++)
			Arrays.fill(rs[i], 0);

		for (int i = 1; i <= n; i++) {
			for (int j = 1; j <= n; j++) {

				if (i != j && A.charAt(i - 1) == A.charAt(j - 1))
					rs[i][j] = rs[i - 1][j - 1] + 1;
				else {
					rs[i][j] = Math.max(rs[i][j - 1], rs[i - 1][j]);
				}

			}
		}

		for (int i = 1; i <= n; i++)
			for (int j = 1; j <= n; j++)
				if (rs[i][j] >= 2)
					return 1;

		return 0;
	}

	public int minPathSum(ArrayList<ArrayList<Integer>> A) {

		int numRows = A.size();
		int numCols = A.get(0).size();

		int minSum[][] = new int[numRows][numCols];

		for (int i = 0; i < numRows; i++) {
			for (int j = 0; j < numCols; j++) {
				if (i == 0 & j == 0)
					minSum[i][j] = A.get(i).get(j);
				else if (i == 0)
					minSum[i][j] = A.get(i).get(j) + minSum[i][j - 1];
				else if (j == 0)
					minSum[i][j] = A.get(i).get(j) + minSum[i - 1][j];
				else
					minSum[i][j] = A.get(i).get(j) + Math.min(minSum[i][j - 1], minSum[i - 1][j]);
			}
		}

		return minSum[numRows - 1][numCols - 1];
	}

	public int isInterleave(String A, String B, String C) {

		if (lcs(C, A) == A.length() && lcs(C, B) == B.length() && A.length() + B.length() == C.length())
			return 1;

		return 0;

	}

	private int lcs(String str1, String str2) {

		int m = str1.length();
		int n = str2.length();

		int lcs[][] = new int[m + 1][n + 1];

		for (int i = 0; i <= m; i++) {
			for (int j = 0; j <= n; j++) {
				if (i == 0 || j == 0)
					lcs[i][j] = 0;
				else if (str1.charAt(i - 1) == str2.charAt(j - 1))
					lcs[i][j] = lcs[i - 1][j - 1] + 1;
				else
					lcs[i][j] = Math.max(lcs[i - 1][j], lcs[i][j - 1]);
			}
		}
		return lcs[m][n];
	}

	// Min Sum Path in Triangle
	public int minimumTotal(ArrayList<ArrayList<Integer>> a) {

		int numRows = a.size();
		int numCols = a.get(numRows - 1).size();

		int minSum[][] = new int[numRows][numCols];

		for (int i = 0; i < numRows; i++)
			Arrays.fill(minSum[i], 0);

		for (int i = 0; i < numRows; i++) {
			for (int j = 0; j < a.get(i).size(); j++) {
				if (i == 0 && j == 0)
					minSum[i][j] = a.get(i).get(j);
				else if (j == 0) {
					minSum[i][j] = a.get(i).get(j) + minSum[i - 1][j];
				} else if (j == a.get(i).size() - 1) {
					minSum[i][j] = a.get(i).get(j) + minSum[i - 1][j];
				} else {
					minSum[i][j] = a.get(i).get(j) + Math.min(minSum[i - 1][j - 1], minSum[i - 1][j + 1]);
				}
			}
		}

		int min = Integer.MAX_VALUE;
		for (int i = 0; i < numCols; i++)
			min = Math.min(min, a.get(numRows - 1).get(i));

		return min;
	}

	public ArrayList<Integer> solve(int R, ArrayList<Integer> S) {

		int kick[] = new int[R + 1];
		ArrayList<ArrayList<Integer>> friends = new ArrayList<>();
		// friends.add(null);
		Arrays.fill(kick, 0);
		int lasti = 0;
		for (int i = 1; i <= R; i++) {
			ArrayList<Integer> kickHist = new ArrayList<>();
			ArrayList<Integer> smallestKickHist = new ArrayList<>();
			for (int j = 0; j < S.size(); j++) {
				if (S.get(j) <= i) {
					if (kick[i] <= S.get(j) + kick[i - S.get(j)]) {
						kick[i] = S.get(j) + kick[i - S.get(j)];
						kickHist.clear();
						if (i - S.get(j) - 1 >= 0)
							kickHist.addAll(friends.get(i - S.get(j) - 1));
						kickHist.add(j);
						if (lasti == i) {
							int prevSum = 0;
							int currSum = 0;
							// check which is lexicographically smaller
							for (int k = 0; k < smallestKickHist.size(); k++)
								prevSum += smallestKickHist.get(k);
							for (int k = 0; k < kickHist.size(); k++)
								currSum += kickHist.get(k);
							if (prevSum > currSum)
								smallestKickHist = kickHist;
						} else {
							smallestKickHist = kickHist;
						}
					}
				}
				// if(i == R) {
				// kickHist.forEach(s -> System.out.println(s));
				// if(kickHist.size() > 0)
				// break;
				// }
				// System.out.println("i: " + i + " kick[i]: " + kick[i]);
				// kickHist.forEach(s -> System.out.println(s));
				// System.out.println();
				lasti = i;
			}
			friends.add(smallestKickHist);
			// kickHist.clear();
		}

		return friends.get(R - 1);
	}

	public int canJump(ArrayList<Integer> A) {

		if (A == null || A.size() == 0)
			return 0;

		if (A.size() == 1)
			return 1;

		if (A.get(0) == 0)
			return 0;

		int minJumps[] = new int[A.size()];
		Arrays.fill(minJumps, 0);
		for (int i = 1; i < A.size(); i++) {
			int minJumpsToI = Integer.MAX_VALUE;
			for (int j = 0; j < i; j++) {
				int maxJumpsFromJ = A.get(j);
				if (j + maxJumpsFromJ >= i && (minJumps[j] != 0 || (j == 0 && maxJumpsFromJ != 0))) {
					minJumpsToI = Math.min(minJumpsToI, minJumps[j] + 1);
				}
			}
			if (minJumpsToI != Integer.MAX_VALUE)
				minJumps[i] = minJumpsToI;
			else
				minJumps[i] = 0;
			System.out.println("minJumps[" + i + "]: " + minJumps[i]);
		}

		return minJumps[A.size() - 1];
	}

	public int canJump2(ArrayList<Integer> A) {
		if (A.size() <= 1)
			return 1;

		// Return -1 if not possible to jump
		if (A.get(0) == 0)
			return 0;

		// initialization
		int maxReach = A.get(0);
		int step = A.get(0);
		int jump = 1;

		// Start traversing array
		for (int i = 1; i < A.size(); i++) {
			// Check if we have reached the end of the array
			if (i == A.size() - 1) {
				return jump != -1 ? 1 : 0;
			}

			// updating maxReach
			maxReach = Math.max(maxReach, i + A.get(i));

			// we used a step to get to the current index
			step--;

			// If no further steps left
			if (step == 0) {
				// we must have used a jump
				jump++;

				// Check if the current index/position or lesser index
				// is the maximum reach point from the previous indexes
				if (i >= maxReach)
					return 0;

				// re-initialize the steps to the amount
				// of steps to reach maxReach from position i.
				step = maxReach - i;
			}
		}

		return 0;
	}

	public int minJump(ArrayList<Integer> A) {

		if (A.size() <= 1)
			return 0;

		if (A.get(0) == 0)
			return 0;

		int minJumps[] = new int[A.size()];
		Arrays.fill(minJumps, 0);
		for (int i = 1; i < A.size(); i++) {
			int minJumpsToI = Integer.MAX_VALUE;
			for (int j = 0; j < i; j++) {
				int maxJumpsFromJ = A.get(j);
				if (j + maxJumpsFromJ >= i && (minJumps[j] != 0 || (j == 0 && maxJumpsFromJ != 0))) {
					minJumpsToI = Math.min(minJumpsToI, minJumps[j] + 1);
				}
			}
			if (minJumpsToI != Integer.MAX_VALUE)
				minJumps[i] = minJumpsToI;
			else
				minJumps[i] = 0;
			System.out.println("minJumps[" + i + "]: " + minJumps[i]);
		}

		return minJumps[A.size() - 1];

	}

	public ArrayList<String> wordBreak(String A, ArrayList<String> B) {
		Set<String> dict = new HashSet<>(B);
		ArrayList<String> result = textContainsWord(A, 0, dict);
		return (result == null ? new ArrayList<String>() : result);
	}

	private ArrayList<String> textContainsWord(String text, int index, Set<String> dict) {

		if (index == text.length()) {
			return new ArrayList<>();
		}

		ArrayList<String> totalWordList = new ArrayList<>();

		for (int i = index + 1; i <= text.length(); i++) {
			String part1 = text.substring(index, i);
			if (dict.contains(part1)) {
				ArrayList<String> words = textContainsWord(text, i, dict);
				if (words != null) {
					if (words.size() == 0)
						totalWordList.add(part1);
					else {
						for (String s : words) {
							// words.set(words.size() - 1, part1 + " " +
							// words.get(words.size() - 1).trim());
							totalWordList.add(part1 + " " + s.trim());
						}
					}
				}
			}
		}

		if (totalWordList.size() == 0)
			return null;
		else
			return totalWordList;
	}

	public int majorityElement(final List<Integer> A) {

		Integer a[] = new Integer[A.size()];
		A.toArray(a);
		Arrays.sort(a);
		int minMajority = (int) Math.floor(a.length / 2);
		for (int i = a.length - 1; i >= minMajority; i--) {
			System.out.println(a[i] + " " + a[i - minMajority]);
			if (a[i].equals(a[i - minMajority])) {
				return a[i];
			} else {
				continue;
			}
		}

		return a[a.length - 1];
	}

	public int seats(String A) {

		List<Integer> personIndex = new LinkedList<>();
		char a[] = new char[A.length()];
		for (int i = 0; i < A.length(); i++)
			a[i] = A.charAt(i);

		// step 1. add all groups to deque
		Deque<int[]> dq = new ArrayDeque<>();

		int i = 0, j = 0, n = a.length;

		while (j < n) {
			// skip '.'
			while (j < n && a[j] == '.')
				j++;
			if (j == n)
				break;
			// go through 'X'
			for (i = j; j < n && a[j] == 'x'; j++) {
			}
			// add group to deque
			dq.addLast(new int[] { i, j - 1 });
		}

		// step 2. merge groups from both ends
		int totalHops = 0;
		int MAX_VALUE = 10000003;

		while (dq.size() > 1) {
			int[] left = dq.peekFirst();
			int[] right = dq.peekLast();

			int lenLeft = left[1] - left[0] + 1;
			int lenRight = right[1] - right[0] + 1;

			if (lenLeft <= lenRight) {
				// merge left two groups
				left = dq.pollFirst();
				totalHops = (totalHops + ((lenLeft * (dq.peekFirst()[0] - left[1] - 1)) % MAX_VALUE) % MAX_VALUE);
				dq.peekFirst()[0] -= lenLeft;
			} else {
				// merge right two groups
				right = dq.pollLast();
				totalHops = (totalHops + ((lenRight * (right[0] - dq.peekLast()[1] - 1)) % MAX_VALUE) % MAX_VALUE);
				dq.peekLast()[1] += lenRight;
			}
		}

		return totalHops % MAX_VALUE;
	}

	public int mice(ArrayList<Integer> M, ArrayList<Integer> H) {

		Collections.sort(M);
		Collections.sort(H);

		int maxMins = Integer.MIN_VALUE;
		for (int i = 0; i < M.size(); i++) {
			maxMins = Math.max(maxMins, Math.abs(M.get(i) - H.get(i)));
		}

		return maxMins;
	}

	public int candy(ArrayList<Integer> C) {

		int sortedRating[] = new int[C.size()];
		Map<Integer, LinkedList<Integer>> valueToIndexMap = new HashMap<>();

		for (int i = 0; i < C.size(); i++) {
			sortedRating[i] = C.get(i);
			if(valueToIndexMap.containsKey(C.get(i))) {
				valueToIndexMap.get(C.get(i)).add(i);
			} else {
				LinkedList<Integer> ll = new LinkedList<>();
				ll.add(i);
				valueToIndexMap.put(C.get(i), ll);
			}
		}

		Arrays.sort(sortedRating);

		int minCandies = 0;
		int lastCandy = 0;
		int candy[] = new int[C.size()];
		Arrays.fill(candy, 0);

		for (int i = 0; i < sortedRating.length; i++) {
			int currIndex = valueToIndexMap.get(sortedRating[i]).pollFirst();
			if (i == 0)
				lastCandy = 1;
			else {
				if (sortedRating[i] == sortedRating[i - 1]) {
					lastCandy = 1;
				} else {
					int max = Math.max(candy[currIndex != 0 ? currIndex - 1 : currIndex],
							candy[currIndex != sortedRating.length - 1 ? currIndex + 1 : currIndex]);
					if (max == 0)
						lastCandy = 1;
					else
						lastCandy = max + 1;
				}
			}
			candy[currIndex] = lastCandy;
			minCandies += lastCandy;

		}

		return minCandies;
	}
	
	public int bulbs(ArrayList<Integer> A) {
    
		int numFlips = 0;
		for(int i = 0 ; i < A.size() ; i++) {
			int currentState = (numFlips % 2 == 0 ? A.get(i) : (A.get(i) == 0 ? 1 : 0));
			if(currentState == 0)
				numFlips++;
		}
		
		return 0;
	}
	
	public int maxp3(ArrayList<Integer> A) {
    
		Collections.sort(A);
		
		int n = A.size();
//		
//		if(A.get(0)*A.get(1) > A.get(n-1)*A.get(n-2))
//			return  A.get(n-1)*A.get(n-2)*A.get(n-3);
//		else {
//			return Math.max(A.get(0)*A.get(1)*A.get(2), A.get(0)*A.get(1)*A.get(n));
//		}
		return Math.max(A.get(n-1)*A.get(n-2)*A.get(n-3), Math.max(A.get(0)*A.get(1)*A.get(2), A.get(0)*A.get(1)*A.get(n-1)));

		
	}

	public int canCompleteCircuit(final List<Integer> G, final List<Integer> C) {
    
		int effectiveGas[] = new int[G.size()];
		
		for(int  i = 0 ; i < G.size() ; i++)
			effectiveGas[i] = G.get(i) - C.get(i);
		
		int gasLeft = 0;
		for(int  i = 0 ; i < effectiveGas.length ; i++)
			gasLeft += effectiveGas[i];
		
		if(gasLeft < 0)
			return -1;
		
		int gasLeftTillI = 0;
		for(int i = 0 ; i < effectiveGas.length ; i++) {
			gasLeftTillI += effectiveGas[i];
			if(effectiveGas[i] < 0)
				continue;
			else {
				int gasLeftStartingFromI = 0;
				int j = i;
				do {
					gasLeftStartingFromI += effectiveGas[j];
					if(gasLeftStartingFromI < 0)
						break;
					if(j == effectiveGas.length - 1)
						j = 0;
					else 
						j++;
				}while (j != i);
				if(gasLeftStartingFromI >= 0)
					return i;
			}
		}
		
		return -1;
		
	}
	
	public static void main(String[] args) {
		DynamicProgramming dp = new DynamicProgramming();
		// System.out.println(dp.numDecodings("2611055971756562"));
		ArrayList<Integer> coins = new ArrayList<>();
		coins.add(1);
		coins.add(2);
		coins.add(3);
		// System.out.println(dp.isScramble("abbbcbaaccacaacc",
		// "acaaaccabcabcbcb"));
		// System.out.println(dp.repeatingSubsequence("abba"));
		Integer a[] = { 45550, 795673, 817297, 463389, 310822, 805076, 920925, 817297, 817297, 182709, 436820, 817297,
				817297, 817297, 817297, 817297, 817297, 263236, 772190, 585741, 354367, 514928, 903332, 817297, 817297,
				222927, 608, 817297, 672384, 817297, 681041, 256380, 805870, 266065, 676851, 817297, 438526, 817297,
				880946, 817297, 817297, 817297, 247558, 817297, 817297, 357701, 839408, 817297, 925734, 817297, 817297,
				798594, 817297, 817297, 735998, 817297, 817297, 817297, 817297, 322939, 817297, 817297, 817297, 812814,
				734999, 817297, 134242, 817297, 118807, 817297, 557107, 156963, 817297, 293811, 817297, 817297, 405032,
				89956, 817297, 4995, 48808, 53643, 817297, 345494, 817297, 817297, 817297, 817297, 996030, 73655,
				917692, 515770, 917138, 817297, 817297, 817297, 817297, 817297, 238606, 817297, 817297, 876877, 817297,
				817297, 817297, 817297, 817297, 817297, 817297, 817297, 759087, 817297, 376711, 817297, 761397, 817297,
				288919, 53522, 817297, 412310, 817297, 323156, 530068, 968505, 703653, 238695, 817297, 817297, 145534,
				142969, 817297, 568236, 817297, 817297, 547877, 817297, 817297, 817297, 159595, 817297, 547748, 587178,
				951289, 829077, 183946, 5238, 525340, 817297, 817297, 817297, 941193, 296599, 817297, 817297, 18742,
				259107, 752688, 817297, 817297, 136640, 249438, 817297, 817297, 817297, 817297, 729092, 817297, 526140,
				709339, 817297, 346760, 868516, 817297, 488457, 817297, 422269, 690955, 817297, 494990, 817297, 817297,
				878670, 152130, 817297, 817297, 817297, 432349, 817297, 817297, 451043, 817297, 606514, 817297, 817297,
				425553, 817297, 273500, 104433, 297289, 801636, 817297, 674138, 895444, 817297, 237525, 817297, 274485,
				980252, 569478, 537620, 817297, 637787, 994649, 980607, 586599, 817297, 552136, 817297, 987194, 15735,
				817297, 817297, 817297, 101278, 817297, 817297, 673307, 231170, 817297, 817297, 817297, 403281, 817297,
				576661, 817297, 583692, 817297, 318073, 817297, 817297, 817297, 967815, 401713, 817297, 911162, 817297,
				796759, 828982, 817297, 763281, 731769, 100773, 817297, 817297, 371010, 926255, 158962, 817297, 904208,
				817297, 235542, 841642, 473435, 817297, 460901, 817297, 782334, 817297, 569957, 728893, 817297, 459182,
				817297, 515670, 817297, 381180, 111842, 817297, 641692, 817297, 349796, 817297, 926385, 950051, 932495,
				817297, 817297, 760357, 4779, 817297, 817297, 730053, 957368, 736526, 429749, 817297, 817297, 817297,
				720715, 817297, 587860, 672037, 470805, 817297, 817297, 817297, 817297, 534528, 107022, 817297, 267143,
				817297, 334588, 783229, 817297, 817297, 817297, 711634, 817297, 163881, 15195, 817297, 817297, 929583,
				817297, 817297, 817297, 817297, 817297, 817297, 397051, 177103, 817297, 673949, 812868, 107051, 817297,
				308762, 817297, 360133, 817297, 457671, 817297, 383996, 817297, 817297, 25696, 105289, 628188, 817297,
				817297, 642160, 565455, 817297, 817297, 817297, 767666, 895213, 817297, 817297, 199610, 817297, 817297,
				817297, 817297, 149061, 817297, 817297, 870323, 817297, 503489, 155123, 817297, 392196, 539414, 41286,
				667807, 232952, 817297, 817297, 30067, 817297, 817297, 50598, 817297, 423805, 817297, 10547, 817297,
				202599, 817297, 512840, 930869, 307292, 972247, 817297, 775489, 817297, 817297, 982806, 817297, 590869,
				817297, 965214, 817297, 817297, 48707, 911713, 795879, 817297, 817297, 677, 362034, 817297, 785684,
				817297, 817297, 109138, 952068, 476727, 817297, 893621, 489639, 817297, 817297, 289458, 538836, 817297,
				817297, 653690, 817297, 817297, 817297, 89647, 817297, 241895, 881263, 432297, 817297, 3041, 18206,
				817297, 672452, 817297, 817297, 65674, 817297, 817297, 817297, 817297, 233272, 817297, 276552, 619920,
				300237, 355936, 817297, 811202, 817297, 817297, 817297, 817297, 192104, 328611, 405959, 518208, 817297,
				835763, 610375, 650213, 817297, 627803, 640230, 817297, 12278, 817297, 817297, 121393, 650577, 912274,
				225163, 309592, 817297, 817297, 817297, 817297, 817297, 817297, 839132, 482902, 817297, 817297, 817297,
				817297, 817297, 817297, 817297, 592996, 817297, 946395, 817297, 817297, 817297, 169558, 382129, 817297,
				817297, 968842, 745267, 972617, 446065, 817297, 817297, 817297, 817297, 375463, 817297, 817297, 30169,
				844394, 817297, 817297, 827693, 449706, 817297, 817297, 650409, 817297, 817297, 667151, 904686, 817297,
				11423, 817297, 342440, 986566, 436543, 817297, 488935, 817297, 61678, 817297, 817297, 159602, 520848,
				855240, 817297, 817297, 817297, 438962, 697510, 817297, 375901, 744587, 817297, 817297, 817297, 54845,
				817297, 564326, 817297, 9167, 817297, 817297, 550286, 232088, 228587, 817297, 817297, 921560, 446717,
				817297, 729103, 376199, 754927, 668547, 804719, 817297, 401162, 817297, 817297, 817297, 930674, 92362,
				817297, 817297, 817297, 280659, 817297, 817297, 349286, 966203, 817297, 817297, 238205, 998091, 201978,
				817297, 817297, 817297, 817297, 517970, 514931, 817297, 817297, 876733, 59568, 440215, 817297, 817297,
				817297, 817297, 201030, 823160, 274353, 817297, 817297, 817297, 817297, 120553, 817297, 817297, 511328,
				194736, 752258, 713055, 817297, 817297, 817297, 655162, 268091, 817297, 817297, 12944, 798699, 817297,
				508366, 817297, 89165, 817297, 835033, 817297, 817297, 817297, 506974, 407170, 338801, 668560, 372551,
				607670, 817297, 795222, 867292, 698203, 576590, 817297, 121896, 817297, 928111, 909304, 500640, 817297,
				817297, 817297, 817297, 817297, 817297, 637466, 817297, 817297, 817297, 325458, 817297, 817297, 55494,
				216918, 74222, 367297, 871278, 878792, 817297, 396708, 817297, 817297, 742484, 957163, 817297, 298271,
				682831, 817297, 817297, 817297, 409713, 817297, 817297, 817297, 817297, 817297, 553618, 817297, 817297,
				817297, 224463, 852006, 971420, 797015, 776982, 790867, 427681, 311461, 752934, 817297, 817297, 817297,
				817380, 817297, 817297, 817297, 957370, 817297, 737503, 817297, 817297, 817297, 817297, 817297, 195089,
				817297, 349020, 363491, 877768, 949619, 872723, 895070, 817297, 819230, 817297, 508772, 817297, 817297,
				988919, 827564, 692408, 817297, 817297, 680604, 159844, 805408, 219151, 817297, 817297, 817297, 817297,
				635061, 281133, 817297, 817297, 817297, 817297, 510461, 817297, 631070, 817297, 817297, 817297, 174599,
				817297, 817297, 746114, 817297, 803001, 817297, 280654, 817297, 894881, 529085, 328922, 738280, 904065,
				817297, 501174, 817297, 817297, 817297, 817297, 817297, 607501, 817297, 941139, 817297, 317590, 817297,
				817297, 817297, 817297, 817297, 531995, 817297, 375321, 817297, 817297, 709186, 817297, 123471, 63145,
				817297, 611178, 689348, 278361, 817297, 19587, 817297, 817297, 817297, 675166, 817297, 817297, 341930,
				817297, 520628, 588666, 468153, 818369, 172489, 817297, 817297, 817297, 912510, 817297, 817297, 75540,
				537448, 664192, 817297, 817297, 485682, 817297, 817297, 817297, 817297, 817297, 473448, 817297, 872999,
				817297, 985949, 817297, 817297, 124204, 817297, 817297, 817297, 678994, 817297, 817297, 635987, 817297,
				277891, 817297, 564490, 695721, 107852, 88145, 817297, 46639, 716194, 817297, 817297, 134441, 780574,
				817297, 658870, 541265, 817297, 817297, 862303, 817297, 817297, 400393, 817297, 803632, 750347, 710469,
				817297, 949874, 817297, 276123, 46874, 719099, 817297, 817297, 457112, 480730, 817297, 552795, 258486,
				817297, 862125, 301792, 414098, 817297, 817297, 817297, 817297, 817297, 817297, 129119, 570988, 817297,
				180510, 817297, 41281, 817297, 189412, 845855, 814219, 817297, 817297, 732951, 260013, 563923, 817297,
				859399, 441392, 817297, 817297, 817297, 817297, 510446, 817297, 925035, 817297, 219460, 817297, 817297,
				883074, 817297, 817297, 817297, 840903, 198162, 296172, 817297, 161981, 817297, 635734, 834334, 817297,
				817297, 817297, 153672, 817297, 992243, 218689, 270024, 723313, 817297, 517357, 817297, 876842, 817297,
				817297, 817297, 227251, 984944, 64521, 817297, 817297, 817297, 817297, 740508, 817297, 884083, 202328,
				341565, 787434, 817297, 955001, 817297, 817297, 13632, 592638, 817297, 35188, 668959, 554733, 817297,
				817297, 341798, 817297, 817297, 817297, 174566, 416959, 817297 };
		// System.out.println(dp.canJump(new
		// ArrayList<Integer>(Arrays.asList(a))));
		// String s[] = {"bababbbb", "bbbabaa", "abbb", "a", "aabbaab", "b",
		// "babaabbbb", "aa", "bb"};
		// System.out.println(dp.wordBreak("aabbbabaaabbbabaabaab",new
		// ArrayList<String>(Arrays.asList(s))));
		// String s[] = {"cat", "and", "sand", "cats", "dog"};
		// System.out.println(dp.wordBreak("catsanddog", new
		// ArrayList<String>(Arrays.asList(s))));
		System.out.println(dp.majorityElement(new ArrayList<Integer>(Arrays.asList(a))));
	}
}
