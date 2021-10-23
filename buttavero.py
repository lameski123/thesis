

class Point:
    counter = 1

    def __init__(self):
        self.name = ""

    def get_closest_point_in_cloud(self, point_cloud):
        Point.counter = Point.counter + 1
        return Point.counter, None



def points2indexes(point_list, point_cloud):

    idxes_list = []

    for item in point_list:
        if isinstance(item, tuple) or isinstance(item, list):
            assert all(isinstance(x, Point) for x in item)
            idxes_list.append((p.get_closest_point_in_cloud(point_cloud)[0] for p in item))

        else:
            idxes_list.append(item.get_closest_point_in_cloud(point_cloud[0]))

    return idxes_list


point_list = [ (Point(), Point()) for _ in range(20)]
print(points2indexes(point_list, None))