import heapq
import random
class LimitedHeap:
    def __init__(self, max_size):
        """
        Initialize a limited-size min heap.
        
        Args:
            max_size (int): Maximum number of elements allowed in the heap
        """
        self.max_size = max_size
        self.heap = []  # List to store heap elements

    def peek_random(self, n = 1):
        """
        Return n random elements from the heap without removing them.
        
        Args:
            n (int): Number of random elements to return
        
        Returns:
            list: List of n random elements from the heap
        """
        if n > len(self.heap):
            raise ValueError("Requested more elements than available in the heap.")
        return random.sample(self.heap, n)    
            
	
    def push(self, item):
        """
        Add an item to the heap. If size exceeds max_size, remove smallest item.
        
        Args:
            item: Item to add to the heap (must be comparable)
        """
        heapq.heappush(self.heap, item)
        
        # If heap size exceeds limit, remove smallest item
        if len(self.heap) > self.max_size:
            heapq.heappop(self.heap)
            
    def pop(self):
        """
        Remove and return the smallest item from the heap.
        
        Returns:
            The smallest item in the heap
            
        Raises:
            IndexError: If heap is empty
        """
        if not self.heap:
            raise IndexError("Heap is empty")
        return heapq.heappop(self.heap)
    
    def peek(self):
        """
        Return the smallest item without removing it.
        
        Returns:
            The smallest item in the heap
            
        Raises:
            IndexError: If heap is empty
        """
        if not self.heap:
            raise IndexError("Heap is empty")
        return self.heap[0]
    
    def size(self):
        """
        Return the current number of items in the heap.
        
        Returns:
            int: Current size of the heap
        """
        return len(self.heap)
    
    def is_empty(self):
        """
        Check if the heap is empty.
        
        Returns:
            bool: True if heap is empty, False otherwise
        """
        return len(self.heap) == 0
    
    def __str__(self):
        """
        Return a string representation of the heap.
        
        Returns:
            str: String representation of heap contents
        """
        return str(self.heap)

# Example usage:
if __name__ == "__main__":
    # Create a heap with maximum size of 3
    limited_heap = LimitedHeap(3)
    
    # Add some numbers
    limited_heap.push(5)
    limited_heap.push(2)
    limited_heap.push(7)
    print("Heap after adding 5, 2, 7:", limited_heap)  # Should show [2, 5, 7]
    
    # Add a number that exceeds limit
    limited_heap.push(1)
    print("Heap after adding 1:", limited_heap)  # Should show [2, 5, 7] (1 pushes out smallest)
    
    # Add a larger number
    limited_heap.push(10)
    print("Heap after adding 10:", limited_heap)  # Should show [5, 10, 7]
    
    # Pop smallest item
    smallest = limited_heap.pop()
    print("Popped smallest:", smallest)  # Should print 5
    print("Heap after pop:", limited_heap)  # Should show [7, 10]