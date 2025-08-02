import cv2
import numpy as np

# 全局变量
points = []  # 存储所有点坐标
origin = None  # 原点
img = None  # 图像变量
base_img = None  # 原始图像副本
scale_factor = 1.0  # 坐标缩放因子（步长）
pixel_size = 0.1  # 像素大小（单位长度/像素）
show_labels = False  # 是否显示坐标标签

# 优化后的字体设置
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.4  # 字体大小
FONT_THICKNESS = 1
AXIS_COLOR = (0, 0, 255)  # 坐标轴颜色 (蓝色)
ORIGIN_COLOR = (255, 255, 255)  # 原点文字颜色 (白色)
POINT_COLOR = (255, 255, 255)  # 点坐标颜色 (白色)
TEXT_BG_COLOR = (0, 0, 0)  # 文本背景颜色 (黑色)
TEXT_BG_ALPHA = 0.5  # 文本背景透明度
POINT_NUMBER_COLOR = (0, 255, 255)  # 点序号颜色 (黄色)


# 带背景的文本绘制函数（添加边界检查和重叠避免）
def draw_text_with_bg(img, text, position, font, font_scale, color, thickness, existing_rects=None):
    # 获取文本大小
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    # 获取图像尺寸
    h, w = img.shape[:2]

    # 计算文本背景位置
    text_x, text_y = position

    # 初始背景框位置
    bg_x1 = text_x - 2
    bg_y1 = text_y - text_size[1] - 2
    bg_x2 = text_x + text_size[0] + 2
    bg_y2 = text_y + 2

    # 如果提供了现有矩形列表，检查重叠并调整位置
    if existing_rects is not None:
        max_attempts = 10
        attempt = 0

        # 检查是否与现有矩形重叠
        def overlaps(rect1, rect2):
            x1, y1, x2, y2 = rect1
            a1, b1, a2, b2 = rect2
            return not (x2 < a1 or a2 < x1 or y2 < b1 or b2 < y1)

        # 尝试找到不重叠的位置
        while attempt < max_attempts:
            overlap_found = False
            current_rect = (bg_x1, bg_y1, bg_x2, bg_y2)

            for rect in existing_rects:
                if overlaps(current_rect, rect):
                    overlap_found = True
                    break

            if not overlap_found:
                break

            # 尝试向下移动
            shift = 15
            text_y += shift
            bg_y1 += shift
            bg_y2 += shift
            attempt += 1

    # 检查并调整位置，确保文本不超出图像边界
    # 检查右边界
    if bg_x2 > w - 5:
        text_x = w - text_size[0] - 5
        bg_x1 = text_x - 2
        bg_x2 = text_x + text_size[0] + 2

    # 检查左边界
    if bg_x1 < 5:
        text_x = 5
        bg_x1 = text_x - 2
        bg_x2 = text_x + text_size[0] + 2

    # 检查下边界
    if bg_y2 > h - 5:
        text_y = h - 5
        bg_y1 = text_y - text_size[1] - 2
        bg_y2 = text_y + 2

    # 检查上边界
    if bg_y1 < 5:
        text_y = text_size[1] + 5
        bg_y1 = text_y - text_size[1] - 2
        bg_y2 = text_y + 2

    # 创建半透明背景
    overlay = img.copy()
    cv2.rectangle(overlay, (int(bg_x1), int(bg_y1)), (int(bg_x2), int(bg_y2)), TEXT_BG_COLOR, -1)
    img = cv2.addWeighted(overlay, TEXT_BG_ALPHA, img, 1 - TEXT_BG_ALPHA, 0)

    # 绘制文本
    cv2.putText(img, text, (int(text_x), int(text_y)), font, font_scale, color, thickness)

    # 返回文本矩形区域和调整后的位置
    return img, (bg_x1, bg_y1, bg_x2, bg_y2)


# 重绘图像函数（包括坐标轴）
def redraw_image():
    global img, base_img, origin, points, scale_factor, pixel_size, show_labels

    # 重置为原始图像
    img = base_img.copy()

    # 存储所有文本矩形区域以避免重叠
    text_rects = []

    # 如果已设置原点，绘制坐标轴
    if origin is not None:
        # 获取图像尺寸
        h, w = img.shape[:2]

        # 绘制坐标轴（十字线）
        cv2.line(img, (0, origin[1]), (w, origin[1]), AXIS_COLOR, 1)  # X轴
        cv2.line(img, (origin[0], 0), (origin[0], h), AXIS_COLOR, 1)  # Y轴

        # 标注坐标轴（添加边界检查）
        # X轴标签
        x_label_pos = (w - 25, origin[1] - 8)
        if origin[1] - 8 < 0:  # 检查上边界
            x_label_pos = (w - 25, origin[1] + 15)
        img, x_rect = draw_text_with_bg(img, "X", x_label_pos,
                                        FONT, FONT_SCALE, POINT_COLOR, FONT_THICKNESS)
        text_rects.append(x_rect)

        # Y轴标签
        y_label_pos = (origin[0] + 8, 25)
        if origin[0] + 8 > w - 30:  # 检查右边界
            y_label_pos = (origin[0] - 30, 25)
        img, y_rect = draw_text_with_bg(img, "Y", y_label_pos,
                                        FONT, FONT_SCALE, POINT_COLOR, FONT_THICKNESS)
        text_rects.append(y_rect)

        # 绘制原点处显示(0,0) - 保留3位小数
        origin_text = f"(0.000, 0.000)"
        origin_pos = (origin[0] + 12, origin[1] - 8)
        if origin[1] - 8 < 0:  # 检查上边界
            origin_pos = (origin[0] + 12, origin[1] + 15)
        img, origin_rect = draw_text_with_bg(img, origin_text, origin_pos,
                                             FONT, FONT_SCALE, ORIGIN_COLOR, FONT_THICKNESS)
        text_rects.append(origin_rect)

        # 绘制所有已记录的点 - 保留3位小数
        for i, point in enumerate(points):
            # 计算相对坐标（应用步长和像素大小）
            dx = (point[0] - origin[0]) * pixel_size * scale_factor
            dy = (point[1] - origin[1]) * pixel_size * scale_factor

            # 在点上绘制序号（黄色小数字）
            cv2.putText(img, str(i + 1), (point[0] + 5, point[1] - 5),
                        FONT, 0.5, POINT_NUMBER_COLOR, 1)

            # 如果标签显示开启
            if show_labels:
                # 显示相对坐标（保留3位小数）
                coord_text = f"({dx:.3f}, {dy:.3f})"

                # 计算文本位置（默认在点右侧）
                text_pos = (point[0] + 15, point[1])

                # 如果点在图像右侧，将文本显示在左侧
                if point[0] > w - 150:  # 如果点在右侧150像素内
                    text_pos = (point[0] - 150, point[1])

                # 如果点在图像底部，将文本显示在上方
                if point[1] > h - 30:
                    text_pos = (text_pos[0], point[1] - 20)

                # 绘制带背景的文本，避免重叠
                img, text_rect = draw_text_with_bg(img, coord_text, text_pos,
                                                   FONT, FONT_SCALE, POINT_COLOR, FONT_THICKNESS, text_rects)
                text_rects.append(text_rect)

    # 显示更新后的图像
    cv2.imshow("Coordinate System", img)


# 鼠标点击事件
def mouse_event(event, x, y, flags, param):
    global origin, points

    if event == cv2.EVENT_LBUTTONDOWN:
        if origin is None:
            # 第一次点击设为原点
            origin = (x, y)
            print(f"原点设置于: ({x:.3f}, {y:.3f})")
        else:
            # 记录其他点
            points.append((x, y))
            dx = (x - origin[0]) * pixel_size * scale_factor
            dy = (y - origin[1]) * pixel_size * scale_factor
            print(f"添加点 {len(points)}: 绝对坐标 ({x:.3f}, {y:.3f}), 相对坐标 ({dx:.3f}, {dy:.3f})")

        # 重绘图像（包括坐标轴）
        redraw_image()


# 设置像素大小（单位长度/像素）
def set_pixel_size():
    global pixel_size
    try:
        value = float(input("请输入像素大小（单位长度/像素，例如0.1）: "))
        if value > 0:
            pixel_size = value
            print(f"像素大小已设置为: {pixel_size:.3f}")
            redraw_image()
        else:
            print("错误: 像素大小必须大于0")
    except ValueError:
        print("错误: 请输入有效的数字")


# 设置步长（缩放因子）
def set_scale_factor():
    global scale_factor
    try:
        value = float(input("请输入步长（缩放因子，例如1.0）: "))
        if value > 0:
            scale_factor = value
            print(f"步长已设置为: {scale_factor:.3f}")
            redraw_image()
        else:
            print("错误: 步长必须大于0")
    except ValueError:
        print("错误: 请输入有效的数字")


# 切换标签显示状态
def toggle_labels():
    global show_labels
    show_labels = not show_labels
    status = "开" if show_labels else "关"
    print(f"坐标标签显示已{status}")
    redraw_image()


# 主程序
if __name__ == "__main__":
    # 替换为你的图片路径
    image_path = "123.png"  # 请修改为实际图片路径
    base_img = cv2.imread(image_path)

    if base_img is None:
        print(f"错误: 无法加载图像 '{image_path}'，请检查文件路径")
        exit()

    img = base_img.copy()

    # 创建窗口并设置回调
    cv2.namedWindow("Coordinate System", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Coordinate System", mouse_event)

    # 显示初始图像
    cv2.imshow("Coordinate System", img)

    print("坐标系设置工具")
    print("操作说明:")
    print("1. 第一次点击: 设置原点(显示坐标轴和(0.000, 0.000))")
    print("2. 后续点击: 在任意位置点击显示相对坐标")
    print("3. 按键说明:")
    print("   - ESC: 退出程序")
    print("   - S: 设置步长（缩放因子）")
    print("   - P: 设置像素大小（单位长度/像素）")
    print("   - H: 显示/隐藏坐标标签")
    print("   - C: 清除所有点（保留原点）")
    print("   - R: 重置整个系统（包括原点）")

    # 等待按键退出
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC键
            break
        elif key == ord('s') or key == ord('S'):  # 设置步长
            set_scale_factor()
        elif key == ord('p') or key == ord('P'):  # 设置像素大小
            set_pixel_size()
        elif key == ord('h') or key == ord('H'):  # 切换标签显示
            toggle_labels()
        elif key == ord('c') or key == ord('C'):  # 清除所有点
            points = []
            print("已清除所有点")
            redraw_image()
        elif key == ord('r') or key == ord('R'):  # 重置整个系统
            origin = None
            points = []
            print("已重置整个坐标系")
            redraw_image()

    cv2.destroyAllWindows()

    # 打印所有坐标结果 - 保留3位小数
    if origin and points:
        print("\n最终坐标结果:")
        print(f"原点坐标: ({origin[0]:.3f}, {origin[1]:.3f})")
        print(f"步长设置: {scale_factor:.3f}, 像素大小: {pixel_size:.3f}")

        for i, point in enumerate(points):
            dx = (point[0] - origin[0]) * pixel_size * scale_factor
            dy = (point[1] - origin[1]) * pixel_size * scale_factor
            print(f"点 {i + 1}: 绝对坐标 ({point[0]:.3f}, {point[1]:.3f}), 相对坐标 ({dx:.3f}, {dy:.3f})")