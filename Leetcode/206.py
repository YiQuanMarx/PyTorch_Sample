# Leetcode 206. Reverse Linked List
# Reverse a singly linked list.
# Example:
# Input: 1->2->3->4->5->NULL
# Output: 5->4->3->2->1->NULL
# Follow up: A linked list can be reversed either iteratively or recursively. Could you implement both?

# Solution:
# Iterative
# Time Complexity: O(n)
# Space Complexity: O(1)
# Runtime: 36 ms, faster than 91.06% of Python3 online submissions for Reverse Linked List.
# Memory Usage: 14.6 MB, less than 5.74% of Python3 online submissions for Reverse Linked List.
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        prev = None
        cur = head
        while cur:
            next_node = cur.next
            cur.next = prev
            prev = cur
            cur = next_node
        return prev