package com.rahul.prep.interviewbit;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
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

	public static void main(String[] args) {
		DynamicProgramming dp = new DynamicProgramming();
		// System.out.println(dp.numDecodings("2611055971756562"));
		ArrayList<Integer> coins = new ArrayList<>();
		coins.add(1);
		coins.add(2);
		coins.add(3);
		System.out.println(dp.isMatch2("cab","ca*"));
	}
}
