def get_node_key(node):
    if hasattr(node, 'op'):
        if node.op == "placeholder":
            return node.target  # e.g., "x"
        elif node.op == "call_function":
            if hasattr(node, 'name') and node.name:
                key = node.name
                print("node name of call_function", key)
                # if key.startswith("%"):
                #     key = key[1:]
                return key  # e.g., "add"
            # else:
            #     key = str(node.target)
            #     print("none name of call_function", key)
            #     if key.startswith("torch.ops."):
            #         key = key[len("torch.ops."):]
            #     return key
    return node


def encode_aten_add_tensor(state, node):
    """
    Encodes aten.add.Tensor to an SMT-based addition op.
    Retrieves SMT expressions for operands using get_node_key and binds the result
    under the call_function node’s key.
    """
    key0 = get_node_key(node.args[0])
    key1 = get_node_key(node.args[1])
    arg0 = state.regs.getExpr(key0)
    arg1 = state.regs.getExpr(key1)
    
    add_expr = arg0 + arg1
    
    # Store result using the call_function node's key.
    state.regs.addExpr(get_node_key(node), add_expr, "Integer")
    
    return add_expr

def encode_aten_mul_tensor(state, node):
    """
    Encodes aten.mul.Tensor to an SMT-based multiplication op.
    """
    key0 = get_node_key(node.args[0])
    key1 = get_node_key(node.args[1])
    arg0 = state.regs.getExpr(key0)
    arg1 = state.regs.getExpr(key1)
    
    mul_expr = arg0 * arg1
    state.regs.addExpr(get_node_key(node), mul_expr, "Integer")
    
    return mul_expr
