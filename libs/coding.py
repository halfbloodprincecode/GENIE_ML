import hashlib

md5 = lambda s: hashlib.md5(s).hexdigest()
sha1 = lambda s: hashlib.sha1(s.encode('utf-8')).hexdigest()
sha256 = lambda s: hashlib.sha256(s.encode('utf-8')).hexdigest()


if __name__ == '__main__':
    # d = {
    #     'name': 'ali',
    #     'ok': 'okkk',
    #     'age': 16
    # }
    # d2 = {
    #     'age': 161,
    #     'name': 'ali1',
    #     'ok': 'okkk1',
    # }
    # s1 = ' | '.join((list(d.keys())))
    # s2 = ' | '.join((list(d2.keys())))
    # print('{} -> {}'.format(s1, sha1(s1)))
    # print('{} -> {}'.format(s2, sha1(s2)))
    # print('='*30)
    # s1 = ' | '.join(set(list(d.keys())))
    # s2 = ' | '.join(set(list(d2.keys())))
    # print('{} -> {}'.format(s1, sha1(s1)))
    # print('{} -> {}'.format(s2, sha1(s2)))

    print(sha1(set(['val/aeloss_step', 'val/discloss_step', 'val/total_loss_step', 'val/quant_loss_step', 'val/nll_loss_step', 'val/rec_loss_step', 'val/p_loss_step', 'val/d_weight_step', 'val/disc_factor_step', 'val/g_loss_step', 'val/disc_loss_step', 'val/logits_real_step', 'val/logits_fake_step', 'epoch'])))