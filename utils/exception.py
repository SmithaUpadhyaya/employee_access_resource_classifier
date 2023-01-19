
class ModuleException(Exception):
    
    def __init__(self, module_name, error):
        
        super().__init__(error)
        self.error = "Error occurred module name: [{0}] error message: [{1}]".format(module_name, error)


    def __str__(self):
        return self.error
