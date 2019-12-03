
def total_variation_norm(input_matrix, beta= 2 ):
        """
            Total variation norm is the second norm in the paper
            represented as R_V(x)
        """
        to_check = input_matrix[:,0, :-1, :-1]  # Trimmed: right - bottom
        one_bottom = input_matrix[:,0, 1:, :-1]  # Trimmed: top - right
        one_right = input_matrix[:,0, :-1, 1:]  # Trimmed: top - right
        total_variation = (((to_check - one_bottom)**2 +
                            (to_check - one_right)**2)**(beta/2)).sum()
        to_check = input_matrix[:,1, :-1, :-1]  # Trimmed: right - bottom
        one_bottom = input_matrix[:,1, 1:, :-1]  # Trimmed: top - right
        one_right = input_matrix[:,1, :-1, 1:]  # Trimmed: top - right
        total_variation += (((to_check - one_bottom)**2 +
                            (to_check - one_right)**2)**(beta/2)).sum()
        to_check = input_matrix[:,2, :-1, :-1]  # Trimmed: right - bottom
        one_bottom = input_matrix[:,2, 1:, :-1]  # Trimmed: top - right
        one_right = input_matrix[:,2, :-1, 1:]  # Trimmed: top - right
        total_variation += (((to_check - one_bottom)**2 +
                            (to_check - one_right)**2)**(beta/2)).sum()
        return  total_variation / (31*32*16)
